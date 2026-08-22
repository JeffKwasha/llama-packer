# tests/test_utils_parsers.py
"""Memory string parsers."""

from __future__ import annotations

import pytest

from llama_packer.utils import parse_context_length, parse_mem_mb


def test_resolve_spare_suffixed_gigabytes():
    assert parse_mem_mb("2G", 32768) == 2048


def test_resolve_spare_suffixed_megabytes():
    assert parse_mem_mb("512m", 32768) == 512


def test_resolve_spare_bare_gb_hint():
    # bare number < 3 * VRAM(GB) -> treated as GB
    assert parse_mem_mb("2", 32768) == 2048


def test_resolve_spare_bare_mb_hint():
    # bare number >= 3 * VRAM(GB) -> treated as MB
    assert parse_mem_mb("512", 32768) == 512


def test_resolve_spare_invalid_returns_zero():
    assert parse_mem_mb("nonsense", 32768) == 0


def test_parse_context_length_k():
    assert parse_context_length("128k") == 131072


def test_parse_context_length_m():
    assert parse_context_length("1m") == 1048576


def test_parse_context_length_bare():
    assert parse_context_length("65536") == 65536
