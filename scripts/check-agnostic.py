"""Acceptance check: the core stays domain-agnostic.

Fails if industry vocabulary appears in the backend core or the UI, or if the engine /
optimizer / persona / validation code compares a domain id (``if domain == "x"`` style
branching). Industry words belong only in domain configs (examples/, backend/app/seed/).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD_DIRS = ["backend/app", "src/pages", "src/components"]
EXCLUDE = {"backend/app/seed"}
BRANCH_DIRS = ["backend/app/engine", "backend/app/optimizer", "backend/app/personas", "backend/app/validation"]

WORDS = re.compile(r"\b(hotel|room|occupancy|loan|gpu|tutor|tour|car|airline)s?\b", re.IGNORECASE)
BRANCH = re.compile(r"\b(domain|domain_id|domain\.id)\s*(==|!=|\bin\b)\s*[\"'\[(]")
SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".json"}
# The UI never prices anything: pages/components may import only display helpers from src/engine.
UI_DIRS = ["src/pages", "src/components"]
UI_ENGINE_IMPORT = re.compile(r"""from\s+["'](?:@/engine|\.\./+engine|\.\./\.\./engine)(?:/(\w+))?["']""")
UI_ALLOWED = {"format", "readback"}


def files(dirs: list[str]):
    for d in dirs:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and p.suffix in SUFFIXES and not any(rel.startswith(e) for e in EXCLUDE) and "__pycache__" not in rel:
                yield p, rel


def main() -> int:
    hits: list[str] = []
    for p, rel in files(WORD_DIRS):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if WORDS.search(line):
                hits.append(f"{rel}:{i}: domain word: {line.strip()}")
    for p, rel in files(BRANCH_DIRS):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if BRANCH.search(line):
                hits.append(f"{rel}:{i}: domain branch: {line.strip()}")
    for p, rel in files(UI_DIRS):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = UI_ENGINE_IMPORT.search(line)
            if m and m.group(1) not in UI_ALLOWED:
                hits.append(f"{rel}:{i}: UI imports pricing code: {line.strip()}")
    if hits:
        print("Found domain-specific code:\n" + "\n".join(hits))
        return 1
    print(f"OK - no domain words in {', '.join(WORD_DIRS)}; no domain branching in {', '.join(BRANCH_DIRS)}; UI imports only display helpers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
