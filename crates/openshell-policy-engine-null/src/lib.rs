// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Null reference engine for the `openshell.policy.v1alpha1.Engine` wire.
//!
//! This crate is **a Gateway-side test fixture**. Its job is to be the
//! simplest possible thing on the other side of a UDS that satisfies the
//! OpenShell-owned `Engine` 4-RPC contract: `Health`, `AcquireHandle`,
//! `GetProjection`, `ReleaseHandle`. It returns a fixed policy projection
//! body loaded at startup and signs the envelope with a dev Ed25519 key.
//!
//! What this is **not**:
//!
//! - It is not the Verifier. There is no bundle pipeline.
//! - It is not RPV. No `Authorize`, no `GetPolicyDigest`, no native
//!   `apf.rpv.v1alpha2` surface.
//! - It is not an attested-policy daemon. There is no trust-root
//!   provisioning, no policy lowering, no canonical-policy decoder.
//!
//! The fixture exists to bridge the OpenShell-side `AttestedPolicyProvider`
//! (W-B) with a real engine process before the RPV-side `Engine`-surface
//! adapter (W-A) is in place. See the APP implementation plan W-A bullet
//! ("In-tree null Verifier") and W-C Phase-B-consequences for the framing.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use ed25519_dalek::{Signer, SigningKey};
use openshell_core::proto::policy::v1alpha1 as wire;
use openshell_server::policy_provider::{
    canonical_projection_bytes, ProjectionEnvelope as GatewayProjectionEnvelope,
};
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};
use wire::engine_server::Engine;

/// Surface id this engine serves. The only one it serves.
pub const SANDBOX_POLICY_SURFACE_ID: &str = "openshell.sandbox.v1";

/// Schema version stamped on every projection envelope this engine emits.
///
/// Lockstep with the gateway-side `AttestedPolicyProvider`'s expectations.
/// The gateway uses `surface_id + schema_version` to decide how to decode
/// `body`; today it routes everything under
/// `openshell.sandbox.v1` through the canonical `SandboxPolicy` decoder.
pub const SANDBOX_POLICY_SCHEMA_VERSION: &str = "v1";

/// Errors raised while loading the projection body at engine startup.
#[derive(Debug, thiserror::Error)]
pub enum ProjectionLoadError {
    #[error("failed to read projection body at '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("projection body at '{path}' has unsupported extension '{ext}' (expected yaml/yml/bin/pb)")]
    UnsupportedExtension { path: String, ext: String },

    #[error("projection body at '{path}' has no extension (expected yaml/yml/bin/pb)")]
    NoExtension { path: String },

    #[error("failed to parse YAML projection body at '{path}': {message}")]
    Yaml { path: String, message: String },

    #[error("failed to decode protobuf projection body at '{path}': {source}")]
    Decode {
        path: String,
        #[source]
        source: prost::DecodeError,
    },
}

/// Load a `SandboxPolicy` projection body from disk and return the
/// protobuf-encoded bytes.
///
/// Format is picked by file extension:
///
/// - `.yaml` / `.yml` → parse via `openshell_policy::parse_sandbox_policy`,
///   then re-encode as protobuf.
/// - `.bin` / `.pb`   → bytes verbatim, after a sanity decode to confirm
///   they are a valid `SandboxPolicy`.
///
/// Other extensions are rejected at startup so a typo never reaches the
/// wire as opaque bytes.
pub fn load_projection_body(path: &Path) -> Result<Vec<u8>, ProjectionLoadError> {
    let bytes = std::fs::read(path).map_err(|source| ProjectionLoadError::Io {
        path: path.display().to_string(),
        source,
    })?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| ProjectionLoadError::NoExtension {
            path: path.display().to_string(),
        })?;

    match ext.as_str() {
        "yaml" | "yml" => {
            let yaml = String::from_utf8(bytes).map_err(|e| ProjectionLoadError::Io {
                path: path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
            })?;
            let policy = openshell_policy::parse_sandbox_policy(&yaml).map_err(|report| {
                ProjectionLoadError::Yaml {
                    path: path.display().to_string(),
                    message: format!("{report:?}"),
                }
            })?;
            Ok(policy.encode_to_vec())
        }
        "bin" | "pb" => {
            // Sanity decode so a malformed .pb fails at startup rather than
            // on the first gateway call.
            openshell_core::proto::SandboxPolicy::decode(bytes.as_slice()).map_err(|source| {
                ProjectionLoadError::Decode {
                    path: path.display().to_string(),
                    source,
                }
            })?;
            Ok(bytes)
        }
        other => Err(ProjectionLoadError::UnsupportedExtension {
            path: path.display().to_string(),
            ext: other.to_string(),
        }),
    }
}

/// SHA-256 fingerprint of an Ed25519 verifying key in lowercase hex. Used
/// in the startup log so a test setup can confirm the trust store carries
/// the matching key.
#[must_use]
pub fn verifying_key_fingerprint(vk: &ed25519_dalek::VerifyingKey) -> String {
    let digest = Sha256::digest(vk.as_bytes());
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Configuration captured at engine startup.
#[derive(Debug, Clone)]
pub struct NullEngineConfig {
    /// `signing_key_id` placed in every `ProjectionEnvelope`. Trust store
    /// on the gateway side keys lookups by this string.
    pub signing_key_id: String,
    /// Pre-loaded protobuf-encoded `SandboxPolicy` bytes returned as
    /// `body` on every `GetProjection`.
    pub projection_body: Vec<u8>,
    /// SHA-256 of `projection_body`. Stamped on every envelope so the
    /// gateway can correlate identical projections across calls.
    pub policy_digest: Vec<u8>,
}

impl NullEngineConfig {
    /// Build a config from a projection body byte slice.
    #[must_use]
    pub fn new(signing_key_id: String, projection_body: Vec<u8>) -> Self {
        let policy_digest = Sha256::digest(&projection_body).to_vec();
        Self {
            signing_key_id,
            projection_body,
            policy_digest,
        }
    }
}

/// In-memory record of a handle the engine has minted.
///
/// The runtime context is captured but otherwise unused — the null engine
/// does not authorize against it, it just remembers enough to honor
/// `ReleaseHandle` and detect "unknown handle" on `GetProjection`.
#[derive(Debug, Clone)]
struct BoundContext {
    sandbox_id: String,
}

/// Null engine state.
///
/// Cloneable so each `tonic::Service` clone shares the handle map and the
/// signing key. The handle map lives behind a Tokio mutex; the signing
/// key is read-only.
#[derive(Debug, Clone)]
pub struct NullEngine {
    config: Arc<NullEngineConfig>,
    signing_key: Arc<SigningKey>,
    handles: Arc<Mutex<HashMap<Vec<u8>, BoundContext>>>,
}

impl NullEngine {
    /// Construct an engine from a pre-loaded config and a signing key.
    #[must_use]
    pub fn new(config: NullEngineConfig, signing_key: SigningKey) -> Self {
        Self {
            config: Arc::new(config),
            signing_key: Arc::new(signing_key),
            handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Number of handles currently tracked. Test-only helper; the public
    /// surface is the gRPC RPCs.
    pub async fn handle_count(&self) -> usize {
        self.handles.lock().await.len()
    }
}

#[async_trait]
impl Engine for NullEngine {
    async fn health(
        &self,
        _req: Request<wire::HealthRequest>,
    ) -> Result<Response<wire::HealthResponse>, Status> {
        // Startup loaded the projection and signing key; if we are alive
        // we are serving. NOT_SERVING is kept as a defensive enum value
        // but is unreachable here.
        Ok(Response::new(wire::HealthResponse {
            status: wire::health_response::Status::Serving as i32,
        }))
    }

    async fn acquire_handle(
        &self,
        req: Request<wire::AcquireHandleRequest>,
    ) -> Result<Response<wire::AcquireHandleResponse>, Status> {
        // The null engine does NOT verify the runtime-context signature.
        // The gateway sets it (see `GrpcPolicySource::acquire_handle`),
        // but this fixture has no trust store for gateway-side keys —
        // adding one would push it past "minimal test fixture" and start
        // duplicating Verifier behavior. Explicit non-goal.
        let envelope = req
            .into_inner()
            .envelope
            .ok_or_else(|| Status::invalid_argument("envelope missing"))?;

        if envelope.sandbox_id.is_empty() {
            return Err(Status::invalid_argument("envelope.sandbox_id is empty"));
        }

        // 16 random bytes. The handle is opaque to the gateway by the
        // wire contract; bytes from `rand` are good enough — this is a
        // test fixture, not a security boundary.
        let handle_bytes: [u8; 16] = rand::random();
        let handle = handle_bytes.to_vec();

        let ctx = BoundContext {
            sandbox_id: envelope.sandbox_id.clone(),
        };
        self.handles.lock().await.insert(handle.clone(), ctx);

        debug!(
            sandbox_id = %envelope.sandbox_id,
            handle_len = handle.len(),
            "null-engine: handle bound"
        );

        Ok(Response::new(wire::AcquireHandleResponse { handle }))
    }

    async fn get_projection(
        &self,
        req: Request<wire::GetProjectionRequest>,
    ) -> Result<Response<wire::GetProjectionResponse>, Status> {
        let inner = req.into_inner();

        if inner.surface_id != SANDBOX_POLICY_SURFACE_ID {
            return Err(Status::failed_precondition(format!(
                "null engine only serves surface '{SANDBOX_POLICY_SURFACE_ID}', got '{}'",
                inner.surface_id
            )));
        }

        // Handle must have been bound. Idempotent release semantics are on
        // ReleaseHandle, not here — an unknown handle on GetProjection is
        // an error.
        {
            let handles = self.handles.lock().await;
            if !handles.contains_key(&inner.handle) {
                return Err(Status::not_found("unknown handle"));
            }
        }

        // Build the envelope in the gateway-internal mirror shape, sign
        // its canonical bytes (the EXACT bytes the AttestedPolicyProvider
        // re-canonicalizes on the verify side via the shared helper),
        // then translate to the wire type for return. Sharing the
        // canonical helper is the load-bearing piece — see the
        // `openshell_server::policy_provider::canonical_projection_bytes`
        // doc comment.
        let mirror = GatewayProjectionEnvelope {
            surface_id: SANDBOX_POLICY_SURFACE_ID.to_string(),
            schema_version: SANDBOX_POLICY_SCHEMA_VERSION.to_string(),
            policy_digest: self.config.policy_digest.clone(),
            // Empty in v0 — the null engine has no bundle pipeline.
            bundle_digest: Vec::new(),
            body: self.config.projection_body.clone(),
            signature: Vec::new(),
            signing_key_id: None,
        };

        let payload = canonical_projection_bytes(&mirror);
        let signature = self.signing_key.sign(&payload).to_bytes().to_vec();

        let envelope = wire::ProjectionEnvelope {
            surface_id: mirror.surface_id,
            schema_version: mirror.schema_version,
            policy_digest: mirror.policy_digest,
            bundle_digest: mirror.bundle_digest,
            body: mirror.body,
            signature,
            signing_key_id: self.config.signing_key_id.clone(),
        };

        Ok(Response::new(wire::GetProjectionResponse {
            envelope: Some(envelope),
        }))
    }

    async fn release_handle(
        &self,
        req: Request<wire::ReleaseHandleRequest>,
    ) -> Result<Response<wire::ReleaseHandleResponse>, Status> {
        let inner = req.into_inner();
        // Contractually idempotent; unknown handle is still OK.
        let removed = self.handles.lock().await.remove(&inner.handle);
        if let Some(ctx) = removed {
            debug!(
                sandbox_id = %ctx.sandbox_id,
                "null-engine: handle released"
            );
        } else {
            debug!(handle_len = inner.handle.len(), "null-engine: release_handle on unknown handle (idempotent)");
        }
        Ok(Response::new(wire::ReleaseHandleResponse {}))
    }
}

/// Helper that logs the startup-summary line every operator wants to see.
pub fn log_startup_summary(
    socket_path: &Path,
    signing_key_id: &str,
    vk: &ed25519_dalek::VerifyingKey,
) {
    info!(
        socket_path = %socket_path.display(),
        signing_key_id,
        fingerprint = %verifying_key_fingerprint(vk),
        surface_id = %SANDBOX_POLICY_SURFACE_ID,
        "null policy engine ready"
    );
}

/// Warn when startup generated a fresh signing key but no
/// `--signing-key-pem` was provided.
///
/// The operator must paste the public key (printed on stdout) into the
/// gateway's trust store for the integration to verify.
pub fn warn_if_ephemeral_key(used_pem: bool) {
    if !used_pem {
        warn!(
            "null engine generated a fresh signing key; copy the PEM printed to stdout into the gateway trust store"
        );
    }
}
