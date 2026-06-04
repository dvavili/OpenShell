// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolves the effective policy for a sandbox through a pluggable driver.
//!
//! [`PolicyResolver`] holds the active [`PolicyDriver`] and the policy
//! surfaces the gateway enforces. [`BuiltinPolicyDriver`] is the default
//! driver and serves policy from the gateway's store.

use crate::persistence::Store;
#[cfg(unix)]
use hyper_util::rt::TokioIo;
use openshell_core::proto::policy_driver::v1::{
    GetCapabilitiesRequest, policy_driver_client::PolicyDriverClient,
};
use openshell_core::proto::{PolicySource, Sandbox, SandboxPolicy};
use std::fmt;
use std::path::Path;
use std::sync::Arc;
#[cfg(unix)]
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint};
use tower::service_fn;

/// Errors returned by a policy driver.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The driver could not produce a policy for the request.
    #[error("policy driver: {0}")]
    Message(String),

    /// The configured driver selection is not yet supported.
    #[error("{0}")]
    Unsupported(String),

    /// Connecting to or handshaking with the configured driver failed.
    #[error("policy driver at '{socket}': {reason}")]
    Connect {
        /// Socket path the gateway tried to reach.
        socket: String,
        /// Underlying transport or handshake error.
        reason: String,
    },

    /// The driver and the gateway share no policy surface.
    #[error(
        "policy driver supports {supported:?} but gateway accepts {accepted:?}; \
         no overlapping surface"
    )]
    NoSurfaceOverlap {
        /// Surfaces the driver reported in its handshake.
        supported: Vec<String>,
        /// Surfaces the gateway is configured to accept.
        accepted: Vec<String>,
    },
}

/// Inputs a driver needs to resolve the policy for one sandbox.
#[derive(Debug, Clone)]
pub struct PolicyRequest<'a> {
    /// The sandbox whose policy is being resolved.
    pub sandbox: &'a Sandbox,
}

impl<'a> PolicyRequest<'a> {
    /// Build a request for a sandbox.
    #[must_use]
    pub fn for_sandbox(sandbox: &'a Sandbox) -> Self {
        Self { sandbox }
    }
}

/// The resolved policy for a sandbox plus the metadata callers report
/// alongside it.
#[derive(Debug, Clone, Default)]
pub struct EffectivePolicy {
    /// The composed policy, or `None` when none is configured.
    pub policy: Option<SandboxPolicy>,
    /// Policy revision number.
    pub version: u32,
    /// Deterministic hash of the composed policy.
    pub policy_hash: String,
    /// Where the policy originated.
    pub policy_source: PolicySource,
    /// Revision number of the active global policy, if any.
    pub global_policy_version: u32,
}

/// A source of sandbox policy.
#[async_trait::async_trait]
pub trait PolicyDriver: Send + Sync {
    /// Driver name, used in logs and audit.
    fn name(&self) -> &str;

    /// Resolve the effective policy and its metadata for the requested
    /// sandbox.
    async fn effective_policy(
        &self,
        request: PolicyRequest<'_>,
    ) -> Result<EffectivePolicy, PolicyError>;

    /// Whether the driver permits the gateway's policy-mutation surface.
    fn permits_mutation(&self) -> bool {
        true
    }
}

/// Default driver. Serves policy from the gateway's store.
#[derive(Clone)]
pub struct BuiltinPolicyDriver {
    store: Arc<Store>,
}

impl fmt::Debug for BuiltinPolicyDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinPolicyDriver")
            .finish_non_exhaustive()
    }
}

impl BuiltinPolicyDriver {
    /// Driver name.
    pub const NAME: &'static str = "builtin";

    /// Build the driver over the gateway's store.
    #[must_use]
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
    }
}

#[async_trait::async_trait]
impl PolicyDriver for BuiltinPolicyDriver {
    fn name(&self) -> &str {
        Self::NAME
    }

    async fn effective_policy(
        &self,
        request: PolicyRequest<'_>,
    ) -> Result<EffectivePolicy, PolicyError> {
        crate::grpc::policy::compose_effective_policy_for_sandbox(
            self.store.as_ref(),
            request.sandbox,
        )
        .await
        .map_err(|status| PolicyError::Message(status.message().to_string()))
    }
}

/// Driver that sources policy from an operator-run process over a unix
/// domain socket.
pub struct ExternalPolicyDriver {
    #[allow(dead_code)]
    client: PolicyDriverClient<Channel>,
    supported_surfaces: Vec<String>,
    permits_mutation: bool,
    driver_version: String,
}

impl fmt::Debug for ExternalPolicyDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalPolicyDriver")
            .field("supported_surfaces", &self.supported_surfaces)
            .field("permits_mutation", &self.permits_mutation)
            .field("driver_version", &self.driver_version)
            .finish_non_exhaustive()
    }
}

impl ExternalPolicyDriver {
    /// Driver name.
    pub const NAME: &'static str = "external";
}

#[async_trait::async_trait]
impl PolicyDriver for ExternalPolicyDriver {
    fn name(&self) -> &str {
        Self::NAME
    }

    async fn effective_policy(
        &self,
        _request: PolicyRequest<'_>,
    ) -> Result<EffectivePolicy, PolicyError> {
        Err(PolicyError::Unsupported(
            "external policy driver projection not yet implemented (RFC 0005 step 2.4)".to_string(),
        ))
    }

    fn permits_mutation(&self) -> bool {
        self.permits_mutation
    }
}

/// Holds the active driver and the policy surfaces the gateway enforces.
#[derive(Clone)]
pub struct PolicyResolver {
    driver: Arc<dyn PolicyDriver>,
    accepted_surfaces: Vec<String>,
}

impl fmt::Debug for PolicyResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolicyResolver")
            .field("driver", &self.driver.name())
            .field("accepted_surfaces", &self.accepted_surfaces)
            .finish_non_exhaustive()
    }
}

impl PolicyResolver {
    /// Build a resolver from a driver and the accepted surfaces.
    #[must_use]
    pub fn new(driver: Arc<dyn PolicyDriver>, accepted_surfaces: Vec<String>) -> Self {
        Self {
            driver,
            accepted_surfaces,
        }
    }

    /// Name of the active driver.
    #[must_use]
    pub fn driver_name(&self) -> &str {
        self.driver.name()
    }

    /// Policy surfaces the gateway enforces.
    #[must_use]
    pub fn accepted_surfaces(&self) -> &[String] {
        &self.accepted_surfaces
    }

    /// The active driver.
    #[must_use]
    pub fn driver(&self) -> &Arc<dyn PolicyDriver> {
        &self.driver
    }

    /// Whether the active driver permits the gateway's policy-mutation
    /// surface.
    #[must_use]
    pub fn permits_mutation(&self) -> bool {
        self.driver.permits_mutation()
    }
}

/// Select the policy driver for the configured surfaces and driver socket.
///
/// `driver_socket` is the driver selector: `None` selects the in-process
/// built-in driver. A configured socket selects a third-party driver, which
/// the caller builds asynchronously via [`connect_external_policy_driver`]; it
/// is rejected here so a synchronous caller never silently falls back to the
/// built-in driver.
pub fn resolve_policy_driver(
    _accepted_surfaces: &[String],
    driver_socket: Option<&Path>,
    store: Arc<Store>,
) -> Result<Arc<dyn PolicyDriver>, PolicyError> {
    match driver_socket {
        None => Ok(Arc::new(BuiltinPolicyDriver::new(store))),
        Some(_) => Err(PolicyError::Unsupported(
            "third-party policy driver requires connect_external_policy_driver".to_string(),
        )),
    }
}

/// Connect to an operator-run policy driver, run the capability handshake, and
/// reconcile its surfaces against the ones the gateway accepts.
///
/// Connects to a socket the operator already created; it never spawns or
/// supervises the driver process. Fails closed on a connection or handshake
/// error and when the driver and gateway share no surface.
#[cfg(unix)]
pub async fn connect_external_policy_driver(
    driver_socket: &Path,
    accepted_surfaces: &[String],
) -> Result<Arc<dyn PolicyDriver>, PolicyError> {
    let socket = driver_socket.to_path_buf();
    let display = socket.display().to_string();

    let connect_socket = socket.clone();
    let channel = Endpoint::from_static("http://[::]:50051")
        .connect_with_connector(service_fn(move |_: tonic::transport::Uri| {
            let socket = connect_socket.clone();
            async move { UnixStream::connect(socket).await.map(TokioIo::new) }
        }))
        .await
        .map_err(|e| PolicyError::Connect {
            socket: display.clone(),
            reason: e.to_string(),
        })?;

    let mut client = PolicyDriverClient::new(channel);
    let capabilities = client
        .get_capabilities(tonic::Request::new(GetCapabilitiesRequest {}))
        .await
        .map_err(|status| PolicyError::Connect {
            socket: display.clone(),
            reason: status.to_string(),
        })?
        .into_inner();

    let supported: Vec<String> = capabilities
        .supported_surfaces
        .iter()
        .filter(|surface| accepted_surfaces.contains(*surface))
        .cloned()
        .collect();
    if supported.is_empty() {
        return Err(PolicyError::NoSurfaceOverlap {
            supported: capabilities.supported_surfaces,
            accepted: accepted_surfaces.to_vec(),
        });
    }

    Ok(Arc::new(ExternalPolicyDriver {
        client,
        supported_surfaces: supported,
        permits_mutation: capabilities.permits_mutation,
        driver_version: capabilities.driver_version,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::test_store;
    use crate::policy_store::PolicyStoreExt;
    use openshell_core::proto::datamodel::v1::ObjectMeta;
    use openshell_core::proto::{Sandbox, SandboxPolicy, SandboxSpec};
    use prost::Message;

    fn sandbox_with_spec(id: &str, policy: Option<SandboxPolicy>) -> Sandbox {
        Sandbox {
            metadata: Some(ObjectMeta {
                id: id.to_string(),
                name: id.to_string(),
                ..Default::default()
            }),
            spec: Some(SandboxSpec {
                policy,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn factory_returns_builtin_driver_without_socket() {
        let store = Arc::new(test_store().await);
        let driver = resolve_policy_driver(&[], None, store).expect("builtin driver");
        assert_eq!(driver.name(), BuiltinPolicyDriver::NAME);
        assert_eq!(driver.name(), "builtin");
    }

    #[tokio::test]
    async fn factory_rejects_configured_driver_socket() {
        let store = Arc::new(test_store().await);
        let socket = Path::new("/run/openshell/policy.sock");
        match resolve_policy_driver(&[], Some(socket), store) {
            Err(PolicyError::Unsupported(_)) => {}
            Ok(_) => panic!("configured socket must fail closed, not run builtin"),
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn coordinator_exposes_accepted_surfaces_and_driver_name() {
        let store = Arc::new(test_store().await);
        let surfaces = vec!["openshell.sandbox.v1".to_string()];
        let resolver = PolicyResolver::new(
            resolve_policy_driver(&surfaces, None, store).expect("builtin driver"),
            surfaces.clone(),
        );
        assert_eq!(resolver.accepted_surfaces(), surfaces.as_slice());
        assert_eq!(resolver.driver_name(), "builtin");
    }

    #[tokio::test]
    async fn coordinator_default_has_no_accepted_surfaces() {
        let store = Arc::new(test_store().await);
        let resolver = PolicyResolver::new(
            resolve_policy_driver(&[], None, store).expect("builtin driver"),
            Vec::new(),
        );
        assert!(resolver.accepted_surfaces().is_empty());
    }

    #[tokio::test]
    async fn builtin_driver_serves_sandbox_spec_policy() {
        let store = Arc::new(test_store().await);
        let policy = SandboxPolicy {
            version: 3,
            ..Default::default()
        };
        let sandbox = Sandbox {
            spec: Some(SandboxSpec {
                policy: Some(policy.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let driver = BuiltinPolicyDriver::new(store);
        let effective = driver
            .effective_policy(PolicyRequest::for_sandbox(&sandbox))
            .await
            .expect("effective policy");
        assert_eq!(effective.policy, Some(policy));
        assert_eq!(effective.version, 1);
        assert_eq!(effective.policy_source, PolicySource::Sandbox);
        assert!(!effective.policy_hash.is_empty());
    }

    #[tokio::test]
    async fn builtin_driver_serves_latest_stored_revision() {
        let store = Arc::new(test_store().await);
        let sandbox = sandbox_with_spec("sb-history", None);

        let stored = SandboxPolicy {
            version: 7,
            ..Default::default()
        };
        store
            .put_policy_revision(
                "rev-3",
                "sb-history",
                3,
                &stored.encode_to_vec(),
                "hash-rev-3",
            )
            .await
            .expect("seed revision");

        let driver = BuiltinPolicyDriver::new(store);
        let effective = driver
            .effective_policy(PolicyRequest::for_sandbox(&sandbox))
            .await
            .expect("effective policy");

        assert_eq!(effective.policy, Some(stored));
        assert_eq!(effective.version, 3);
        assert_eq!(effective.policy_hash, "hash-rev-3");
        assert_eq!(effective.policy_source, PolicySource::Sandbox);
    }

    #[tokio::test]
    async fn builtin_driver_backfills_spec_policy_as_revision_one() {
        let store = Arc::new(test_store().await);
        let policy = SandboxPolicy {
            version: 4,
            ..Default::default()
        };
        let sandbox = sandbox_with_spec("sb-backfill", Some(policy.clone()));

        assert!(
            store
                .get_latest_policy("sb-backfill")
                .await
                .expect("query history")
                .is_none()
        );

        let driver = BuiltinPolicyDriver::new(store.clone());
        let effective = driver
            .effective_policy(PolicyRequest::for_sandbox(&sandbox))
            .await
            .expect("effective policy");

        assert_eq!(effective.version, 1);
        assert_eq!(effective.policy, Some(policy.clone()));

        let backfilled = store
            .get_latest_policy("sb-backfill")
            .await
            .expect("query history")
            .expect("revision one is written");
        assert_eq!(backfilled.version, 1);
        assert_eq!(backfilled.policy_hash, effective.policy_hash);
        let decoded = SandboxPolicy::decode(backfilled.policy_payload.as_slice())
            .expect("decode backfilled policy");
        assert_eq!(decoded, policy);
    }

    #[tokio::test]
    async fn builtin_driver_returns_empty_when_no_policy() {
        let store = Arc::new(test_store().await);
        let sandbox = sandbox_with_spec("sb-empty", None);

        let driver = BuiltinPolicyDriver::new(store);
        let effective = driver
            .effective_policy(PolicyRequest::for_sandbox(&sandbox))
            .await
            .expect("effective policy");

        assert_eq!(effective.policy, None);
        assert_eq!(effective.version, 0);
        assert!(effective.policy_hash.is_empty());
        assert_eq!(effective.policy_source, PolicySource::Sandbox);
    }

    #[cfg(unix)]
    mod external {
        use super::*;
        use openshell_core::proto::policy_driver::v1::policy_driver_server::{
            PolicyDriver as PolicyDriverService, PolicyDriverServer,
        };
        use openshell_core::proto::policy_driver::v1::{
            AcquireHandleRequest, AcquireHandleResponse, GetCapabilitiesResponse,
            GetProjectionRequest, GetProjectionResponse, ReleaseHandleRequest,
            ReleaseHandleResponse,
        };
        use std::path::PathBuf;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::net::UnixListener;
        use tonic::transport::Server;
        use tonic::{Request, Response, Status};

        struct CapabilitiesDouble {
            supported_surfaces: Vec<String>,
            permits_mutation: bool,
        }

        #[tonic::async_trait]
        impl PolicyDriverService for CapabilitiesDouble {
            async fn get_capabilities(
                &self,
                _request: Request<GetCapabilitiesRequest>,
            ) -> Result<Response<GetCapabilitiesResponse>, Status> {
                Ok(Response::new(GetCapabilitiesResponse {
                    supported_surfaces: self.supported_surfaces.clone(),
                    permits_mutation: self.permits_mutation,
                    driver_version: "double-0".to_string(),
                }))
            }

            async fn acquire_handle(
                &self,
                _request: Request<AcquireHandleRequest>,
            ) -> Result<Response<AcquireHandleResponse>, Status> {
                Err(Status::unimplemented("acquire_handle"))
            }

            async fn get_projection(
                &self,
                _request: Request<GetProjectionRequest>,
            ) -> Result<Response<GetProjectionResponse>, Status> {
                Err(Status::unimplemented("get_projection"))
            }

            async fn release_handle(
                &self,
                _request: Request<ReleaseHandleRequest>,
            ) -> Result<Response<ReleaseHandleResponse>, Status> {
                Err(Status::unimplemented("release_handle"))
            }
        }

        fn unique_socket_path(test_name: &str) -> PathBuf {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be after unix epoch")
                .as_nanos();
            PathBuf::from(format!(
                "/tmp/openshell-policy-{test_name}-{}-{nanos}.sock",
                std::process::id()
            ))
        }

        fn spawn_driver(
            socket: &Path,
            supported_surfaces: Vec<String>,
            permits_mutation: bool,
        ) -> tokio::task::JoinHandle<()> {
            let _ = std::fs::remove_file(socket);
            let listener = UnixListener::bind(socket).expect("test socket should bind");
            let double = CapabilitiesDouble {
                supported_surfaces,
                permits_mutation,
            };
            tokio::spawn(async move {
                let incoming = futures::stream::unfold(listener, |listener| async move {
                    let item = listener.accept().await.map(|(stream, _)| stream);
                    Some((item, listener))
                });
                let _ = Server::builder()
                    .add_service(PolicyDriverServer::new(double))
                    .serve_with_incoming(incoming)
                    .await;
            })
        }

        #[tokio::test]
        async fn connects_and_reconciles_overlapping_surface() {
            let socket = unique_socket_path("overlap");
            let server = spawn_driver(
                &socket,
                vec![
                    "openshell.sandbox.v1".to_string(),
                    "vendor.other.v1".to_string(),
                ],
                true,
            );

            let accepted = vec!["openshell.sandbox.v1".to_string()];
            let driver = connect_external_policy_driver(&socket, &accepted)
                .await
                .expect("driver connects");
            assert_eq!(driver.name(), "external");
            assert_eq!(driver.name(), ExternalPolicyDriver::NAME);
            assert!(driver.permits_mutation());

            let resolver = PolicyResolver::new(driver, accepted.clone());
            assert!(resolver.permits_mutation());

            server.abort();
            let _ = std::fs::remove_file(&socket);
        }

        #[tokio::test]
        async fn fails_closed_on_no_overlap() {
            let socket = unique_socket_path("no-overlap");
            let server = spawn_driver(&socket, vec!["vendor.other.v1".to_string()], true);

            let accepted = vec!["openshell.sandbox.v1".to_string()];
            match connect_external_policy_driver(&socket, &accepted).await {
                Err(PolicyError::NoSurfaceOverlap { .. }) => {}
                Ok(_) => panic!("expected no-overlap error, driver connected"),
                Err(other) => panic!("expected no-overlap error, got {other:?}"),
            }

            server.abort();
            let _ = std::fs::remove_file(&socket);
        }

        #[tokio::test]
        async fn fails_closed_on_empty_accepted_surfaces() {
            let socket = unique_socket_path("empty-accepted");
            let server = spawn_driver(&socket, vec!["openshell.sandbox.v1".to_string()], true);

            match connect_external_policy_driver(&socket, &[]).await {
                Err(PolicyError::NoSurfaceOverlap { .. }) => {}
                Ok(_) => panic!("expected no-overlap error, driver connected"),
                Err(other) => panic!("expected no-overlap error, got {other:?}"),
            }

            server.abort();
            let _ = std::fs::remove_file(&socket);
        }

        #[tokio::test]
        async fn fails_closed_on_unreachable_socket() {
            let socket = unique_socket_path("unreachable");
            let _ = std::fs::remove_file(&socket);

            let accepted = vec!["openshell.sandbox.v1".to_string()];
            match connect_external_policy_driver(&socket, &accepted).await {
                Err(PolicyError::Connect { .. }) => {}
                Ok(_) => panic!("expected connect error, driver connected"),
                Err(other) => panic!("expected connect error, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn stores_permits_mutation_false_from_handshake() {
            let socket = unique_socket_path("no-mutation");
            let server = spawn_driver(&socket, vec!["openshell.sandbox.v1".to_string()], false);

            let accepted = vec!["openshell.sandbox.v1".to_string()];
            let driver = connect_external_policy_driver(&socket, &accepted)
                .await
                .expect("driver connects");
            assert!(!driver.permits_mutation());

            let resolver = PolicyResolver::new(driver, accepted);
            assert!(!resolver.permits_mutation());

            server.abort();
            let _ = std::fs::remove_file(&socket);
        }
    }
}
