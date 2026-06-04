// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolves the effective policy for a sandbox through a pluggable driver.
//!
//! [`PolicyResolver`] holds the active [`PolicyDriver`] and the policy
//! surfaces the gateway enforces. [`BuiltinPolicyDriver`] is the default
//! driver and serves policy from the gateway's store.

use crate::persistence::Store;
use openshell_core::proto::{PolicySource, Sandbox, SandboxPolicy};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Errors returned by a policy driver.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The driver could not produce a policy for the request.
    #[error("policy driver: {0}")]
    Message(String),

    /// The configured driver selection is not yet supported.
    #[error("{0}")]
    Unsupported(String),
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
}

/// Select the policy driver for the configured surfaces and driver socket.
///
/// `driver_socket` is the driver selector: `None` selects the in-process
/// built-in driver; `Some(_)` selects a third-party driver at that socket and
/// fails closed until that path is implemented. A configured socket never
/// silently falls back to the built-in driver.
pub fn resolve_policy_driver(
    _accepted_surfaces: &[String],
    driver_socket: Option<&Path>,
    store: Arc<Store>,
) -> Result<Arc<dyn PolicyDriver>, PolicyError> {
    match driver_socket {
        None => Ok(Arc::new(BuiltinPolicyDriver::new(store))),
        Some(_) => Err(PolicyError::Unsupported(
            "third-party policy driver not yet implemented (RFC 0005 step 2.3)".to_string(),
        )),
    }
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
}
