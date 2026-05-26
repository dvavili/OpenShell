// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! APF Runtime Policy Verifier (RPV) shadow integration.
//!
//! Gateway-side client for the v1alpha2 wire surface described in the
//! OpenShell **Attested Policy Projection** RFC. When configured (see
//! [`RpvShadow::from_env`]), the gateway makes shadow calls into the
//! local RPV daemon during sandbox lifecycle events:
//!
//!   - Probe `Health()` at startup; fail-closed on anything other than OK.
//!   - For each sandbox admission, build and sign a `RuntimeContextEnvelope`
//!     using the gateway's dedicated runtime-context signing key
//!     (separate from the sandbox-JWT key — RFC §Runtime-context
//!     attestation, trust-role separation), then call
//!     `BindRuntimeContext`.
//!   - Immediately call `GetProjection(handle, "openshell.substrate.v1")`
//!     to fetch the projection the substrate would consume.
//!
//! Shadow-only: the projection is logged, not enforced. The integration
//! is opt-in per deployment via env vars; when unset, the gateway behaves
//! exactly as today.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use ed25519_dalek::SigningKey;
use rpv_client::{
    RpvClientHandle, build_signed_envelope,
    proto::{HealthState, RuntimeContextEnvelope},
};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Env var pointing at the RPV daemon's UDS socket. When unset, the
/// integration is fully disabled.
const ENV_SOCKET: &str = "OPENSHELL_RPV_SOCKET";

/// Env var pointing at the file containing the gateway's runtime-context-
/// envelope signing key (Ed25519, `ed25519:<base64-of-32-byte-seed>`
/// format). The matching public key is provisioned into the RPV daemon's
/// `--gateway-pubkey` at deployment time. Required when [`ENV_SOCKET`] is
/// set.
const ENV_SIGNING_KEY: &str = "OPENSHELL_RPV_GATEWAY_SIGNING_KEY";

/// Env var holding the user subject the gateway attests on every
/// runtime-context envelope. On Omnistation this is a static
/// per-workspace fact established at portal provisioning time
/// (workspace-to-user binding). When unset, the gateway falls back to
/// `"unknown@workspace.local"` and logs a warning.
const ENV_USER_SUBJECT: &str = "OPENSHELL_RPV_USER_SUBJECT";

/// Env var that, when set to a truthy value (`1`, `true`, `yes`),
/// switches the integration from shadow to **enforce**. In enforce
/// mode:
///   - A failed Health probe at gateway startup is fatal — the
///     gateway refuses to come up.
///   - A failed `BindRuntimeContext` / `GetProjection` at sandbox
///     admission returns an error to the CreateSandbox caller, refusing
///     admission.
/// When unset/false the integration runs in shadow mode: failures are
/// logged and sandbox creation proceeds (the default and current safe
/// rollout posture).
const ENV_ENFORCE: &str = "OPENSHELL_RPV_ENFORCE";

/// Env var pointing at a directory to write each admission's
/// projection bytes into. When set, every successful
/// `shadow_admit_sandbox` writes `<dir>/<sandbox_id>.yaml` with the
/// projection RPV vended. Optional — used for inspection / debugging
/// of what RPV would return per sandbox.
const ENV_DUMP_PROJECTIONS_DIR: &str = "OPENSHELL_RPV_DUMP_PROJECTIONS_DIR";

const OPENSHELL_SURFACE_ID: &str = "openshell.substrate.v1";

/// Configured shadow integration. Constructed at gateway startup; if any
/// required input is missing, the constructor returns `Ok(None)` and the
/// gateway runs without RPV calls.
pub struct RpvShadow {
    socket_path: PathBuf,
    signing_key: SigningKey,
    user_subject: String,
    enforce: bool,
    dump_projections_dir: Option<PathBuf>,
    client: Mutex<Option<RpvClientHandle>>,
}

impl std::fmt::Debug for RpvShadow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpvShadow")
            .field("socket_path", &self.socket_path)
            .field("user_subject", &self.user_subject)
            .field("enforce", &self.enforce)
            .field("dump_projections_dir", &self.dump_projections_dir)
            .finish_non_exhaustive()
    }
}

impl RpvShadow {
    /// Construct from environment. Returns `Ok(None)` when the
    /// integration is not configured (i.e. [`ENV_SOCKET`] is unset);
    /// otherwise returns a configured shadow or an error if required
    /// inputs are missing/invalid.
    pub fn from_env() -> Result<Option<Arc<Self>>, RpvShadowError> {
        let Some(socket) = std::env::var_os(ENV_SOCKET) else {
            return Ok(None);
        };
        let socket_path = PathBuf::from(&socket);

        let key_path = std::env::var_os(ENV_SIGNING_KEY).ok_or_else(|| {
            RpvShadowError::Misconfigured(format!(
                "{ENV_SOCKET} is set but {ENV_SIGNING_KEY} is not"
            ))
        })?;
        let signing_key = load_signing_key(Path::new(&key_path))?;

        let user_subject = std::env::var(ENV_USER_SUBJECT).unwrap_or_else(|_| {
            warn!(
                "{ENV_USER_SUBJECT} not set; using fallback subject. \
                 Subject_binding checks against signed bundles will fail."
            );
            "unknown@workspace.local".to_string()
        });

        let enforce = std::env::var(ENV_ENFORCE)
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        let dump_projections_dir = std::env::var_os(ENV_DUMP_PROJECTIONS_DIR).map(PathBuf::from);
        if let Some(dir) = &dump_projections_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                return Err(RpvShadowError::Misconfigured(format!(
                    "{ENV_DUMP_PROJECTIONS_DIR}={}: create_dir_all failed: {e}",
                    dir.display()
                )));
            }
        }

        Ok(Some(Arc::new(Self {
            socket_path,
            signing_key,
            user_subject,
            enforce,
            dump_projections_dir,
            client: Mutex::new(None),
        })))
    }

    /// Whether the integration is configured to fail-closed on RPV
    /// failure. Callers (e.g. the sandbox-create handler) should refuse
    /// admission when an admission attempt errors and this is true;
    /// log-and-continue when this is false.
    #[must_use]
    pub fn enforce(&self) -> bool {
        self.enforce
    }

    /// Probe the RPV daemon's health. Called at gateway startup; the
    /// gateway aborts admission if this returns Err or anything other
    /// than `HealthState::Ok`.
    pub async fn probe_health(&self) -> Result<(), RpvShadowError> {
        let mut client = self.connected_client().await?;
        let resp = client.health().await.map_err(RpvShadowError::Rpv)?;
        let state = HealthState::try_from(resp.state).unwrap_or(HealthState::Unspecified);
        info!(
            socket = %self.socket_path.display(),
            user_subject = %self.user_subject,
            state = ?state,
            "rpv-shadow: health probe"
        );
        if state != HealthState::Ok {
            return Err(RpvShadowError::Unhealthy(state));
        }
        Ok(())
    }

    /// Run the BindRuntimeContext + GetProjection shadow flow for a
    /// freshly-created sandbox. Logs each step; returns the projection's
    /// `source_bundle_digest` so the caller can stamp it on audit events.
    pub async fn shadow_admit_sandbox(
        &self,
        sandbox_id: &str,
    ) -> Result<RpvAdmission, RpvShadowError> {
        let mut client = self.connected_client().await?;

        let envelope: RuntimeContextEnvelope =
            build_signed_envelope(&self.signing_key, sandbox_id, &self.user_subject);
        info!(
            sandbox_id,
            user_subject = %self.user_subject,
            claims_bytes = envelope.claims.len(),
            sig_bytes = envelope.signature.len(),
            "rpv-shadow: built signed runtime-context envelope"
        );

        let bind = client
            .bind_runtime_context(envelope)
            .await
            .map_err(RpvShadowError::Rpv)?;
        if !bind.ok {
            warn!(
                sandbox_id,
                reasons = ?bind.rejection_reasons,
                "rpv-shadow: BindRuntimeContext rejected"
            );
            return Err(RpvShadowError::BindRejected(bind.rejection_reasons));
        }
        info!(sandbox_id, handle = %bind.handle, "rpv-shadow: bound");

        let proj = client
            .get_projection(&bind.handle, OPENSHELL_SURFACE_ID)
            .await
            .map_err(RpvShadowError::Rpv)?;
        if !proj.ok {
            warn!(
                sandbox_id,
                handle = %bind.handle,
                reasons = ?proj.rejection_reasons,
                "rpv-shadow: GetProjection rejected"
            );
            return Err(RpvShadowError::GetProjectionRejected(proj.rejection_reasons));
        }
        let projection_sha256 = {
            let mut h = Sha256::new();
            h.update(&proj.projection);
            hex::encode(h.finalize())
        };
        info!(
            sandbox_id,
            handle = %bind.handle,
            source_bundle_digest = %proj.source_bundle_digest,
            projection_sha256 = %projection_sha256,
            surface_id = %proj.surface_id,
            schema = %proj.projection_schema_version,
            projection_bytes = proj.projection.len(),
            "rpv-shadow: projection vended"
        );

        if let Some(dir) = &self.dump_projections_dir {
            let dump_path = dir.join(format!("{sandbox_id}.yaml"));
            match std::fs::write(&dump_path, &proj.projection) {
                Ok(()) => info!(
                    sandbox_id,
                    path = %dump_path.display(),
                    bytes = proj.projection.len(),
                    "rpv-shadow: projection dumped"
                ),
                Err(e) => warn!(
                    sandbox_id,
                    path = %dump_path.display(),
                    error = %e,
                    "rpv-shadow: projection dump failed (admission still succeeds)"
                ),
            }
        }

        Ok(RpvAdmission {
            handle: bind.handle,
            source_bundle_digest: proj.source_bundle_digest,
            projection_sha256,
            projection_bytes: proj.projection,
        })
    }

    /// Get or establish the cached client connection.
    async fn connected_client(&self) -> Result<RpvClientHandle, RpvShadowError> {
        let mut guard = self.client.lock().await;
        if guard.is_none() {
            let client = RpvClientHandle::connect(&self.socket_path)
                .await
                .map_err(RpvShadowError::Rpv)?;
            *guard = Some(client);
        }
        Ok(guard.as_ref().unwrap().clone())
    }
}

/// Outcome of a successful shadow admission. The handle and digest are
/// captured for audit correlation; the projection bytes are what
/// OpenShell would consume in enforce mode.
#[derive(Debug, Clone)]
pub struct RpvAdmission {
    pub handle: String,
    /// SHA-256 of the *source bundle* (lowercase hex). Returned by the
    /// Verifier; ties projection back to a specific signed bundle.
    pub source_bundle_digest: String,
    /// SHA-256 of the *projection bytes themselves* (lowercase hex).
    /// Computed gateway-side over `projection_bytes`. Changes whenever
    /// the daemon vends different content — useful for spotting bundle
    /// rotations (where the bundle digest changes) vs in-daemon
    /// projection swaps (where only this digest changes).
    pub projection_sha256: String,
    pub projection_bytes: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum RpvShadowError {
    #[error("rpv-shadow misconfigured: {0}")]
    Misconfigured(String),
    #[error("rpv-shadow signing key {path}: {reason}")]
    SigningKeyInvalid { path: PathBuf, reason: String },
    #[error("rpv-shadow rpc error: {0}")]
    Rpv(#[from] rpv_client::RpvError),
    #[error("rpv-shadow daemon unhealthy: state={0:?}")]
    Unhealthy(HealthState),
    #[error("rpv-shadow BindRuntimeContext rejected: {0:?}")]
    BindRejected(Vec<String>),
    #[error("rpv-shadow GetProjection rejected: {0:?}")]
    GetProjectionRejected(Vec<String>),
}

fn load_signing_key(path: &Path) -> Result<SigningKey, RpvShadowError> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| RpvShadowError::SigningKeyInvalid {
            path: path.to_path_buf(),
            reason: format!("read failed: {e}"),
        })?;
    let line = raw
        .lines()
        .find(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .ok_or_else(|| RpvShadowError::SigningKeyInvalid {
            path: path.to_path_buf(),
            reason: "file is empty".to_string(),
        })?
        .trim();
    let b64 = line
        .strip_prefix("ed25519:")
        .ok_or_else(|| RpvShadowError::SigningKeyInvalid {
            path: path.to_path_buf(),
            reason: "expected `ed25519:` prefix".to_string(),
        })?;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| RpvShadowError::SigningKeyInvalid {
            path: path.to_path_buf(),
            reason: format!("base64 decode: {e}"),
        })?;
    if raw.len() != 32 {
        return Err(RpvShadowError::SigningKeyInvalid {
            path: path.to_path_buf(),
            reason: format!("expected 32-byte seed, got {}", raw.len()),
        });
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&raw);
    Ok(SigningKey::from_bytes(&seed))
}
