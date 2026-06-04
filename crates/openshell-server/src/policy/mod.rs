// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolves the effective policy for a sandbox through a pluggable driver.
//!
//! [`PolicyResolver`] holds the active [`PolicyDriver`] and the policy
//! surfaces the gateway enforces. [`BuiltinPolicyDriver`] is the default
//! driver and serves policy from the gateway's store.

use openshell_core::proto::{Sandbox, SandboxPolicy};
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

/// A source of sandbox policy.
#[async_trait::async_trait]
pub trait PolicyDriver: Send + Sync {
    /// Driver name, used in logs and audit.
    fn name(&self) -> &str;

    /// Resolve the policy for the requested sandbox, or `None` when none is
    /// configured.
    async fn effective_policy(
        &self,
        request: PolicyRequest<'_>,
    ) -> Result<Option<SandboxPolicy>, PolicyError>;
}

/// Default driver. Serves policy from the gateway's store.
#[derive(Debug, Default, Clone)]
pub struct BuiltinPolicyDriver;

impl BuiltinPolicyDriver {
    /// Driver name.
    pub const NAME: &'static str = "builtin";

    /// Build the driver.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl PolicyDriver for BuiltinPolicyDriver {
    fn name(&self) -> &str {
        Self::NAME
    }

    async fn effective_policy(
        &self,
        _request: PolicyRequest<'_>,
    ) -> Result<Option<SandboxPolicy>, PolicyError> {
        // Policy composition is handled in the sandbox-config request path.
        Err(PolicyError::Message(
            "builtin policy composition is handled in the sandbox-config path".into(),
        ))
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
pub fn resolve_policy_driver(_accepted_surfaces: &[String]) -> Arc<dyn PolicyDriver> {
    Arc::new(BuiltinPolicyDriver::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_returns_builtin_driver() {
        let driver = resolve_policy_driver(&[]);
        assert_eq!(driver.name(), BuiltinPolicyDriver::NAME);
        assert_eq!(driver.name(), "builtin");
    }

    #[test]
    fn coordinator_exposes_accepted_surfaces_and_driver_name() {
        let surfaces = vec!["openshell.sandbox.v1".to_string()];
        let resolver = PolicyResolver::new(resolve_policy_driver(&surfaces), surfaces.clone());
        assert_eq!(resolver.accepted_surfaces(), surfaces.as_slice());
        assert_eq!(resolver.driver_name(), "builtin");
    }

    #[test]
    fn coordinator_default_has_no_accepted_surfaces() {
        let resolver = PolicyResolver::new(resolve_policy_driver(&[]), Vec::new());
        assert!(resolver.accepted_surfaces().is_empty());
    }
}
