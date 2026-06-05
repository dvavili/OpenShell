// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use openshell_core::proto::policy_driver::v1::{
    Projection, policy_driver_client::PolicyDriverClient, policy_driver_server::PolicyDriver,
};
use prost::Message;

#[test]
fn projection_round_trips_with_signature_and_audit_context() {
    let mut audit_context = std::collections::HashMap::new();
    audit_context.insert("source_digest".to_string(), "sha256:abc".to_string());
    audit_context.insert("request_id".to_string(), "req-42".to_string());

    let original = Projection {
        surface_id: "openshell.sandbox.v1".to_string(),
        policy_version: 7,
        policy_digest: "deadbeef".to_string(),
        body: vec![1, 2, 3, 4],
        signature: Some(vec![9, 8, 7]),
        signing_key_id: Some("key-1".to_string()),
        audit_context,
    };

    let encoded = original.encode_to_vec();
    let decoded = Projection::decode(encoded.as_slice()).expect("decode projection");

    assert_eq!(original, decoded);
}

#[test]
fn projection_round_trips_without_signature_fields() {
    let original = Projection {
        surface_id: "openshell.sandbox.v1".to_string(),
        policy_version: 0,
        policy_digest: "deadbeef".to_string(),
        body: vec![1, 2, 3, 4],
        signature: None,
        signing_key_id: None,
        audit_context: std::collections::HashMap::new(),
    };

    let encoded = original.encode_to_vec();
    let decoded = Projection::decode(encoded.as_slice()).expect("decode projection");

    assert_eq!(decoded.signature, None);
    assert_eq!(decoded.signing_key_id, None);
    assert_eq!(original, decoded);
}

// The generated client and server stubs resolve at the documented path.
#[allow(dead_code)]
fn _type_resolves<C, S: PolicyDriver>(_client: PolicyDriverClient<C>) {}
