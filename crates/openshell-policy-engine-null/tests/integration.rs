// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test between the OpenShell-side
//! `AttestedPolicyProvider` and the null engine over a real UDS.
//!
//! This is the load-bearing deliverable for the null-engine session: if
//! the canonical-byte ordering on the engine side does not match the
//! gateway-side verify path, this test fails with a signature-verify
//! error rather than compiling-but-passing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ed25519_dalek::pkcs8::EncodePublicKey;
use ed25519_dalek::SigningKey;
use openshell_core::proto::policy::v1alpha1::engine_server::EngineServer;
use openshell_policy_engine_null::{NullEngine, NullEngineConfig};
use openshell_server::policy_provider::{
    AttestedPolicyProvider, GrpcPolicySource, PolicyProvider, TrustStore,
};
use prost::Message;
use rand_core_06::OsRng;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

/// Spawn the null engine on a freshly-bound UDS and return the socket path,
/// a join handle for the server, and a shutdown sender.
struct EngineFixture {
    socket_path: PathBuf,
    _server: JoinHandle<()>,
    shutdown: tokio::sync::oneshot::Sender<()>,
}

impl EngineFixture {
    fn spawn(engine: NullEngine, socket_dir: &tempfile::TempDir) -> Self {
        let socket_path = socket_dir.path().join("engine.sock");
        // Bind before returning so the client never races with a not-yet-
        // listening socket.
        let listener = UnixListener::bind(&socket_path).expect("bind UDS");
        let incoming = UnixListenerStream::new(listener);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(EngineServer::new(engine))
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("null engine serve");
        });

        Self {
            socket_path,
            _server: server,
            shutdown: shutdown_tx,
        }
    }

    fn shutdown(self) {
        let _ = self.shutdown.send(());
    }
}

/// Helper: load a `SandboxPolicy` from a YAML fragment and produce the
/// protobuf bytes the null engine would otherwise load from disk.
fn projection_bytes_from_yaml(yaml: &str) -> Vec<u8> {
    let policy = openshell_policy::parse_sandbox_policy(yaml).expect("parse YAML");
    policy.encode_to_vec()
}

/// Helper: build a JSON trust store carrying a single key id → PEM mapping.
fn write_trust_store(
    dir: &tempfile::TempDir,
    key_id: &str,
    vk: &ed25519_dalek::VerifyingKey,
) -> PathBuf {
    let pem = vk
        .to_public_key_pem(ed25519_dalek::pkcs8::spki::der::pem::LineEnding::LF)
        .expect("encode public-key PEM");
    let json = format!(
        r#"{{"keys":[{{"key_id":{key_id:?},"public_key_pem":{pem:?}}}]}}"#,
    );
    let path = dir.path().join("trust.json");
    std::fs::write(&path, json).expect("write trust store");
    path
}

/// Wait for the UDS to be reachable. The server is spawned on a Tokio task
/// so there is a brief window where the file exists but `serve_with_incoming`
/// has not yet started its accept loop. A tight retry loop is sufficient.
async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..50 {
        if tokio::net::UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("null engine socket {} never became reachable", path.display());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attested_provider_resolves_policy_from_null_engine() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let vk = signing_key.verifying_key();
    let key_id = "k-null-it-1";

    let projection = projection_bytes_from_yaml("version: 1\n");
    let config = NullEngineConfig::new(key_id.to_string(), projection.clone());
    let engine = NullEngine::new(config, signing_key);

    let socket_dir = tempfile::tempdir().expect("tmp dir");
    let trust_dir = tempfile::tempdir().expect("tmp dir");

    let fixture = EngineFixture::spawn(engine, &socket_dir);
    wait_for_socket(&fixture.socket_path).await;

    let trust_path = write_trust_store(&trust_dir, key_id, &vk);
    let trust_store = TrustStore::load(&trust_path).expect("load trust store");

    let source = GrpcPolicySource::connect(&fixture.socket_path)
        .await
        .expect("connect to null engine over UDS");
    let provider = AttestedPolicyProvider::new(Arc::new(source), trust_store)
        .await
        .expect("attested provider constructs (health passes)");

    let policy = provider
        .get_effective_policy("test-sandbox")
        .await
        .expect("policy fetch ok")
        .expect("policy present");

    // Round-trip check: the policy the gateway sees must equal the one the
    // engine was configured with. Any mismatch in canonical-byte ordering
    // would have surfaced as a signature-verify error before we got here,
    // so reaching this assertion already proves the two wires meet.
    let expected = openshell_core::proto::SandboxPolicy::decode(projection.as_slice())
        .expect("decode expected policy");
    assert_eq!(policy.version, expected.version);

    fixture.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attested_provider_rejects_when_signing_key_id_is_not_trusted() {
    // The null engine signs with `k-engine`; the trust store knows only
    // `k-other` (a different key). The gateway must refuse the envelope.
    let signing_key = SigningKey::generate(&mut OsRng);
    let other_key = SigningKey::generate(&mut OsRng);

    let projection = projection_bytes_from_yaml("version: 2\n");
    let engine_config = NullEngineConfig::new("k-engine".to_string(), projection);
    let engine = NullEngine::new(engine_config, signing_key);

    let socket_dir = tempfile::tempdir().expect("tmp dir");
    let trust_dir = tempfile::tempdir().expect("tmp dir");

    let fixture = EngineFixture::spawn(engine, &socket_dir);
    wait_for_socket(&fixture.socket_path).await;

    // Trust store carries a different key id than the one the engine
    // stamps onto the envelope.
    let trust_path = write_trust_store(&trust_dir, "k-other", &other_key.verifying_key());
    let trust_store = TrustStore::load(&trust_path).expect("load trust store");

    let source = GrpcPolicySource::connect(&fixture.socket_path)
        .await
        .expect("connect");
    let provider = AttestedPolicyProvider::new(Arc::new(source), trust_store)
        .await
        .expect("attested provider constructs");

    let err = provider
        .get_effective_policy("test-sandbox")
        .await
        .expect_err("untrusted signing_key_id must reject");

    // Surface the rejection as a SourceError(Rejected) so callers can
    // distinguish "engine unreachable" from "engine reachable but
    // verify failed". The exact reason string mentions the unknown key id.
    let msg = format!("{err}");
    assert!(
        msg.contains("k-engine") || msg.to_lowercase().contains("unknown"),
        "expected rejection to mention unknown key id, got: {msg}"
    );

    fixture.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn null_engine_health_serves_immediately() {
    // Defensive: Health returns SERVING the moment the engine is running.
    // Catches regressions where startup ordering shifts and Health blocks.
    let signing_key = SigningKey::generate(&mut OsRng);
    let projection = projection_bytes_from_yaml("version: 1\n");
    let engine = NullEngine::new(
        NullEngineConfig::new("k-1".into(), projection),
        signing_key,
    );

    let socket_dir = tempfile::tempdir().expect("tmp dir");
    let fixture = EngineFixture::spawn(engine, &socket_dir);
    wait_for_socket(&fixture.socket_path).await;

    let trust_dir = tempfile::tempdir().expect("tmp dir");
    let any_key = SigningKey::generate(&mut OsRng);
    let trust_path = write_trust_store(&trust_dir, "k-1", &any_key.verifying_key());
    let trust_store = TrustStore::load(&trust_path).expect("load");

    let source = GrpcPolicySource::connect(&fixture.socket_path)
        .await
        .expect("connect");

    // `AttestedPolicyProvider::new` runs `source.health()` internally; a
    // successful construction is the health assertion.
    let _provider = AttestedPolicyProvider::new(Arc::new(source), trust_store)
        .await
        .expect("attested provider constructs (health reported SERVING)");

    fixture.shutdown();
}
