"""Golden-set format.

```yaml
cases:
  - id: L01
    group: policies              # report grouping
    user: alice                  # passed to your adapter as-is (ACL / tenant), optional
    question: How fast must a data breach be reported?
    relevant:                    # when is a hit relevant? any rule may match; {} = nothing is (abstain case)
      title_prefix: ["Incident protocol"]
      ids: ["doc-17"]
      metadata: {tenant: T1}
    facts: ["72 hours", ["DPO", "data protection officer"]]   # string, or list of equivalent phrasings
    reference: The authority must be notified within 72 hours.  # gold answer for judge metrics
    forbidden_terms: ["Contoso"]   # must not appear in the answer (unless the question itself names it)
    forbidden_metadata: {tenant: [T2]}   # hits with these values are leaks for this case
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ragevalkit.text import normalize


@dataclass
class Case:
    id: str
    question: str
    group: str = "default"
    user: Any = None
    relevant: dict = field(default_factory=dict)
    facts: list = field(default_factory=list)
    reference: str = ""
    forbidden_terms: list[str] = field(default_factory=list)
    forbidden_metadata: dict[str, list] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)

    def is_relevant(self, hit) -> bool:
        rel = self.relevant
        if any(hit.title.startswith(p) for p in rel.get("title_prefix", [])):
            return True
        if hit.id in rel.get("ids", []):
            return True
        meta = rel.get("metadata") or {}
        return bool(meta) and all(hit.metadata.get(k) == v for k, v in meta.items())

    def is_forbidden_hit(self, hit) -> bool:
        return any(hit.metadata.get(k) in values for k, values in self.forbidden_metadata.items())

    def terms_that_must_not_appear(self) -> list[str]:
        """Forbidden terms minus those the user named in the question (echoing them is not a leak)."""
        q = normalize(self.question)
        return [t for t in self.forbidden_terms if normalize(t) not in q]


KNOWN = set(Case.__dataclass_fields__) - {"extra"}


def load_cases(path: str | Path) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cases = []
    for entry in raw["cases"]:
        known = {k: v for k, v in entry.items() if k in KNOWN}
        extra = {k: v for k, v in entry.items() if k not in KNOWN}
        cases.append(Case(**known, extra=extra))
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case ids: {sorted(dupes)}")
    return cases
