# tests/test_persist.py
"""Sidecar fit-params persistence preserves comments and body."""

from __future__ import annotations

from llama_packer.vram import FitParams


def test_persist_preserves_comments_and_body(make_model, tmp_path):
    model = make_model("p", context_length=32768)
    md = tmp_path / "p.md"
    md.write_text(
        "---\n"
        "name: p\n"
        "# keep this frontmatter comment\n"
        "context_length: 32768\n"
        "---\n"
        "\n"
        "# body heading\n"
        "Some markdown body.\n"
    )

    model.vram._persist(FitParams(1000, 0.5, 100, "fit-params", "q8_0", 1))

    content = md.read_text()
    assert "keep this frontmatter comment" in content
    assert "# body heading" in content
    assert "Some markdown body." in content
    assert "fit-params" in content
    assert "model_mib: 1000" in content
