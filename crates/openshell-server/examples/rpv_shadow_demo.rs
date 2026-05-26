// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end demo of the OpenShell-side `rpv_shadow` integration against
//! a running RPV daemon. Proves the OpenShell Gateway can:
//!   1. Connect to RPV over UDS.
//!   2. Probe Health and fail-closed on anything but OK.
//!   3. Mint and sign a runtime-context envelope using its dedicated
//!      runtime-context signing key.
//!   4. Call BindRuntimeContext and receive a handle.
//!   5. Call GetProjection(handle, "openshell.substrate.v1") and receive
//!      the projection the substrate would consume.
//!
//! Run with:
//!     OPENSHELL_RPV_SOCKET=/tmp/rpv-demo.sock \
//!     OPENSHELL_RPV_GATEWAY_SIGNING_KEY=/path/to/gateway-pv-dev.key \
//!     OPENSHELL_RPV_USER_SUBJECT=dvavili@nvidia.com \
//!     cargo run --release --example rpv_shadow_demo -p openshell-server

use openshell_server::rpv_shadow::RpvShadow;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let shadow = RpvShadow::from_env()?
        .ok_or_else(|| anyhow::anyhow!("OPENSHELL_RPV_SOCKET is unset; nothing to demo"))?;

    tracing::info!("openshell-side: probing rpv health");
    shadow.probe_health().await?;

    let sandbox_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("demo-sb-{}", std::process::id()));

    tracing::info!(sandbox_id = %sandbox_id, "openshell-side: shadow admission");
    let admission = shadow.shadow_admit_sandbox(&sandbox_id).await?;

    tracing::info!(
        sandbox_id = %sandbox_id,
        handle = %admission.handle,
        source_bundle_digest = %admission.source_bundle_digest,
        projection_bytes = admission.projection_bytes.len(),
        "openshell-side: shadow admission OK"
    );

    // Optionally print the first 120 chars of the projection so a reader
    // can see it's the OpenShell-native substrate YAML.
    let preview = String::from_utf8_lossy(&admission.projection_bytes);
    let head: String = preview.chars().take(240).collect();
    tracing::info!(projection_head = %head, "openshell-side: projection preview");

    Ok(())
}
