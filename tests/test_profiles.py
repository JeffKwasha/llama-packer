"""profiles.yaml key validation."""

from __future__ import annotations

import logging

from llama_packer.profiles import validate_profiles_keys


def test_unknown_profiles_key_warns_with_a_hint(caplog):
    # Validates: docs/plans/cpu-residency-budget.md #29 -- a top-level key
    # nothing reads warns (with a near-miss hint) instead of doing nothing.
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING):
        validate_profiles_keys({"device_role": ["embeddings"],
                                "usage": {}}, log)
    msgs = [r.message for r in caplog.records]
    assert any("unknown key 'device_role'" in m
               and "did you mean 'device_roles'" in m for m in msgs)
    assert not any("unknown key 'usage'" in m for m in msgs)
