"""Retrieval evaluation — deterministic, no LLM, no API key. Suitable as a CI gate.

Per case, on exactly what your system hands to the LLM:
  hit_at_k        a relevant hit is among the top k
  precision_at_k  relevant hits / min(k, relevant docs in corpus)   (normalised: 1/1 is reachable)
  mrr             1 / rank of the first relevant hit
  fact_recall     share of the case's facts that appear in the LLM context
  leaks           hits the user must not see (adapter's can_see + case.forbidden_metadata) — must be 0

`fact_recall` is the metric that found the case study's main bug: retrieval was perfect,
but the context was truncated before the facts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from ragevalkit.dataset import Case
from ragevalkit.system import RAGSystem
from ragevalkit.text import fact_share

GATED = ("hit_at_k", "precision_at_k", "mrr", "fact_recall")


@dataclass
class CaseScore:
    id: str
    group: str
    hit_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    fact_recall: float | None
    leaks: int
    top: list[str]


def score_case(system: RAGSystem, case: Case, k: int, corpus=None) -> CaseScore:
    r = system.retrieve(case)
    hits = r.hits[:k]
    leaks = sum(1 for h in hits if not system.can_see(case, h) or case.is_forbidden_hit(h))
    hit = prec = rr = None
    if case.answerable:
        flags = [case.is_relevant(h) for h in hits]
        hit = float(any(flags))
        n_rel = sum(1 for d in corpus if case.is_relevant(d)) if corpus else k
        prec = sum(flags) / min(k, n_rel) if n_rel else 0.0
        rr = next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    return CaseScore(case.id, case.group, hit, prec, rr, fact_share(case.facts, "\n".join(r.contexts)),
                     leaks, [h.title or h.id for h in hits])


def aggregate(scores: list[CaseScore]) -> dict:
    def avg(attr: str, subset: list[CaseScore]) -> float | None:
        vals = [getattr(s, attr) for s in subset if getattr(s, attr) is not None]
        return round(mean(vals), 4) if vals else None

    out: dict = {m: avg(m, scores) for m in GATED}
    out["leaks"] = sum(s.leaks for s in scores)
    out["n_cases"] = len(scores)
    out["by_group"] = {g: {m: avg(m, [s for s in scores if s.group == g]) for m in GATED}
                       for g in sorted({s.group for s in scores})}
    return out


def evaluate(system: RAGSystem, cases: list[Case], k: int = 5) -> tuple[dict, list[CaseScore]]:
    corpus = system.corpus()
    scores = [score_case(system, c, k, corpus) for c in cases]
    return aggregate(scores), scores


def compare(current: dict, baseline: dict, tolerance: float = 0.02) -> list[str]:
    """Regressions vs. a stored baseline. Any leak is always a regression."""
    problems = []
    if current["leaks"]:
        problems.append(f"leaks: {current['leaks']} (allowed: 0)")
    for m in GATED:
        cur, base = current.get(m), baseline.get(m)
        if cur is not None and base is not None and cur < base - tolerance:
            problems.append(f"{m}: {cur:.4f} < baseline {base:.4f} (tolerance {tolerance})")
    return problems


def report(summary: dict, scores: list[CaseScore], k: int) -> str:
    f = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [f"### Retrieval · top_k={k}", "", "| group | hit@k | precision@k | MRR | facts in context |",
             "|---|---:|---:|---:|---:|"]
    for g, m in summary["by_group"].items():
        lines.append(f"| {g} | {f(m['hit_at_k'])} | {f(m['precision_at_k'])} | {f(m['mrr'])} | {f(m['fact_recall'])} |")
    lines.append(f"| **all** | **{f(summary['hit_at_k'])}** | **{f(summary['precision_at_k'])}** | "
                 f"**{f(summary['mrr'])}** | **{f(summary['fact_recall'])}** |")
    lines += ["", f"leaks: **{summary['leaks']}** · cases: {summary['n_cases']}", "",
              "| id | hit | MRR | facts | top-1 |", "|---|---:|---:|---:|---|"]
    lines += [f"| {s.id} | {f(s.hit_at_k)} | {f(s.mrr)} | {f(s.fact_recall)} | {s.top[0] if s.top else '–'} |"
              for s in scores]
    return "\n".join(lines)


def save(path: Path, summary: dict, scores: list[CaseScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "cases": [asdict(s) for s in scores]}, indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
