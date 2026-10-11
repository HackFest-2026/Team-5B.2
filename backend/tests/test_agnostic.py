"""The domain-independence check passes on the real code and catches violations."""

from __future__ import annotations

import importlib.util

from tests.conftest import ROOT

spec = importlib.util.spec_from_file_location("check_agnostic", ROOT / "scripts" / "check-agnostic.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)  # type: ignore[union-attr]


def test_core_is_domain_agnostic() -> None:
    assert check.main() == 0


def test_patterns_catch_domain_words_and_branches() -> None:
    assert check.WORDS.search("price per Room")
    assert check.WORDS.search("if gpus > 3")
    assert not check.WORDS.search("cart scarcity carry")
    assert check.BRANCH.search('if domain == "x":')
    assert check.BRANCH.search("if domain_id in ['a']:")
    assert not check.BRANCH.search("if request.domain_id != domain.id:")
