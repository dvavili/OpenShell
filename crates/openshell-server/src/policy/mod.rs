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
use std::sync::Arc;

/// Errors returned by a policy driver.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The driver could not produce a policy for the request.
    #[error("policy driver: {0}")]
    Message(String),
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

/// Select the policy driver for the given accepted surfaces.
#[must_use]
pub fn resolve_policy_driver(
    _accepted_surfaces: &[String],
    store: Arc<Store>,
) -> Arc<dyn PolicyDriver> {
    Arc::new(BuiltinPolicyDriver::new(store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::test_store;
    use openshell_core::proto::{Sandbox, SandboxPolicy, SandboxSpec};

    #[tokio::test]
    async fn factory_returns_builtin_driver() {
        let store = Arc::new(test_store().await);
        let driver = resolve_policy_driver(&[], store);
        assert_eq!(driver.name(), BuiltinPolicyDriver::NAME);
        assert_eq!(driver.name(), "builtin");
    }

    #[tokio::test]
    async fn coordinator_exposes_accepted_surfaces_and_driver_name() {
        let store = Arc::new(test_store().await);
        let surfaces = vec!["openshell.sandbox.v1".to_string()];
        let resolver =
            PolicyResolver::new(resolve_policy_driver(&surfaces, store), surfaces.clone());
        assert_eq!(resolver.accepted_surfaces(), surfaces.as_slice());
        assert_eq!(resolver.driver_name(), "builtin");
    }

    #[tokio::test]
    async fn coordinator_default_has_no_accepted_surfaces() {
        let store = Arc::new(test_store().await);
        let resolver = PolicyResolver::new(resolve_policy_driver(&[], store), Vec::new());
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
}
