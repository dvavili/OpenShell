// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Null policy engine binary — the Gateway-side test fixture that
//! satisfies the `openshell.policy.v1alpha1.Engine` 4-RPC wire.
//!
//! See `openshell_policy_engine_null` library docs for what this is and
//! is not.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ed25519_dalek::pkcs8::spki::der::pem::LineEnding;
use ed25519_dalek::pkcs8::{DecodePrivateKey, EncodePublicKey};
use ed25519_dalek::SigningKey;
use openshell_core::proto::policy::v1alpha1::engine_server::EngineServer;
use openshell_policy_engine_null::{
    load_projection_body, log_startup_summary, warn_if_ephemeral_key, NullEngine,
    NullEngineConfig,
};
use rand_core_06::OsRng;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    name = "openshell-policy-engine-null",
    about = "Null reference Engine for the openshell.policy.v1alpha1 wire (test fixture only)"
)]
struct Cli {
    /// UDS path to bind. Parent directory must exist; path must not be in
    /// use.
    #[arg(long)]
    socket_path: PathBuf,

    /// Path to a `SandboxPolicy` projection body. Extension `.yaml`/`.yml`
    /// parses as the canonical `SandboxPolicy` YAML; `.bin`/`.pb` is taken
    /// verbatim as protobuf-encoded bytes.
    #[arg(long)]
    projection_body: PathBuf,

    /// Value stamped into every `ProjectionEnvelope.signing_key_id`. The
    /// gateway-side trust store keys lookups by this string.
    #[arg(long)]
    signing_key_id: String,

    /// Optional path to an Ed25519 private key in PKCS#8 PEM form. When
    /// absent, a fresh signing key is generated and its public-key PEM is
    /// printed to stdout for the operator to copy into the gateway trust
    /// store.
    #[arg(long)]
    signing_key_pem: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    // -- Pre-flight checks for the UDS path.
    //
    // The connector on the gateway side will retry, so a clean startup
    // failure here is preferable to silently shadowing an existing
    // socket.
    if cli.socket_path.exists() {
        return Err(anyhow!(
            "socket path '{}' already exists; remove it before starting",
            cli.socket_path.display()
        ));
    }
    if let Some(parent) = cli.socket_path.parent() {
        // Empty parent (relative bare filename) is OK — bind in cwd.
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(anyhow!(
                "socket path parent directory '{}' does not exist",
                parent.display()
            ));
        }
    }

    // -- Load the projection body once at startup. A bad body fails fast.
    let projection_body = load_projection_body(&cli.projection_body)
        .with_context(|| format!("loading {}", cli.projection_body.display()))?;
    info!(
        path = %cli.projection_body.display(),
        bytes = projection_body.len(),
        "loaded projection body"
    );

    // -- Resolve the signing key.
    let (signing_key, used_pem) = if let Some(p) = cli.signing_key_pem.as_ref() {
        let pem = std::fs::read_to_string(p)
            .with_context(|| format!("reading signing key PEM at {}", p.display()))?;
        let sk = SigningKey::from_pkcs8_pem(&pem)
            .map_err(|e| anyhow!("invalid PKCS#8 PEM at {}: {e}", p.display()))?;
        (sk, true)
    } else {
        let sk = SigningKey::generate(&mut OsRng);
        let pem = sk
            .verifying_key()
            .to_public_key_pem(LineEnding::LF)
            .map_err(|e| anyhow!("encoding generated public key as PEM: {e}"))?;
        // Print to stdout so a test harness can pipe / copy. Logs go to
        // stderr, so the two streams stay separable.
        println!("# null policy engine — generated public key (PEM)");
        println!("# key_id: {}", cli.signing_key_id);
        print!("{pem}");
        (sk, false)
    };

    warn_if_ephemeral_key(used_pem);

    let vk = signing_key.verifying_key();
    log_startup_summary(&cli.socket_path, &cli.signing_key_id, &vk);

    let config = NullEngineConfig::new(cli.signing_key_id.clone(), projection_body);
    let engine = NullEngine::new(config, signing_key);

    // -- Bind the UDS and serve.
    //
    // Mirrors the tonic UDS server pattern used elsewhere in the
    // workspace (see openshell-server compute drivers).
    let listener = UnixListener::bind(&cli.socket_path).with_context(|| {
        format!("binding UDS at {}", cli.socket_path.display())
    })?;
    let incoming = UnixListenerStream::new(listener);

    // Ensure we clean up the socket on shutdown so a restart does not
    // require manual cleanup.
    let socket_path_for_cleanup = cli.socket_path.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("ctrl-c received; shutting down null engine");
    };

    let result = Server::builder()
        .add_service(EngineServer::new(engine))
        .serve_with_incoming_shutdown(incoming, shutdown)
        .await;

    // Best-effort socket cleanup.
    let _ = std::fs::remove_file(&socket_path_for_cleanup);

    result.context("null engine server exited with error")
}
