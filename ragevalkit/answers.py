"""Answer evaluation through your system's real retrieve + generate path.

Metrics (grade keys; null where not applicable):
  fact_coverage        share of the case's facts in the answer               programmatic
  terms_safe           no forbidden term in the answer                        programmatic
  faithfulness_de      answer claims supported by the context                 DeepEval  (strict, see below)
  faithfulness_ragas   same, independent implementation                       RAGAS
  relevancy_de         answer addresses the question                          DeepEval
  context_recall_ragas reference answer derivable from the context            RAGAS
  abstention           abstains instead of inventing, for unanswerable cases  Claude judge

Two faithfulness implementations on purpose: their agreement is your judge calibration, and
disagreement marks cases worth reading. DeepEval runs with penalize_ambiguous_claims=True — its
default counts only CONTRADICTIONS as unfaithful, so a claim absent from the context scores as
faithful (the case study's oracle check caught this).

Modes: app (default) · oracle (answer = case.reference) · null (empty answer) · copy (answer =
first context). Run oracle/null/copy BEFORE trusting a comparison: they test the graders.

Output per variant: results.jsonl (one row per case, written as it completes), traces/<id>_rep<k>.json,
errors.jsonl (harness/API failures — never scored as a model zero). Resumes per (id, rep).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from statistics import mean

from ragevalkit.dataset import Case
from ragevalkit.judge import ClaudeJudge, JudgeError, deepeval_model, judge_abstention, ragas_llm
from ragevalkit.system import Answer, RAGSystem
from ragevalkit.text import fact_share, normalize

METRICS = ["fact_coverage", "faithfulness_de", "faithfulness_ragas", "relevancy_de",
           "context_recall_ragas", "terms_safe", "abstention"]
CASE_TIMEOUT_S = 600


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def programmatic(case: Case, response: str) -> dict:
    norm = normalize(response)
    return {"fact_coverage": fact_share(case.facts, response),
            "terms_safe": float(not any(normalize(t) in norm for t in case.terms_that_must_not_appear()))}


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def judged(case: Case, response: str, contexts: list[str], judge: ClaudeJudge,
           frameworks: set[str], judge_reps: int = 1) -> tuple[dict, dict]:
    """Judge metrics, each averaged over `judge_reps` independent passes.

    Judge passes are not deterministic: RAGAS first rewrites the answer into statements and may
    add entities that were never there ("you must …" -> "the data controller must …"), which a
    strict NLI step then rejects. The same answer scored 0.0 and 1.0 on two passes in the toy
    example. Averaging passes trades cost for a stable number.
    """
    scores = {k: None for k in METRICS if k not in ("fact_coverage", "terms_safe")}
    why: dict = {}
    if not case.answerable:
        v = judge_abstention(judge, case.question, response)
        scores["abstention"] = float(v.abstains and not v.invents_specifics)
        why["abstention"] = v.reason
        return scores, why
    if not response.strip():
        scores["relevancy_de"] = 0.0  # faithfulness of an empty answer would be a vacuous 1.0 — not scored
        return scores, why
    ctx = contexts or ["(no context)"]
    if "deepeval" in frameworks:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        model = deepeval_model(judge)
        tc = LLMTestCase(input=case.question, actual_output=response, retrieval_context=ctx)
        for key, make in (("faithfulness_de", lambda: FaithfulnessMetric(model=model, async_mode=False,
                                                                           penalize_ambiguous_claims=True)),
                          ("relevancy_de", lambda: AnswerRelevancyMetric(model=model, async_mode=False))):
            passes = []
            for _ in range(judge_reps):
                metric = make()
                metric.measure(tc)
                if metric.error:
                    raise JudgeError(f"{key}: {metric.error}")
                passes.append(float(metric.score))
                why[key] = metric.reason
            scores[key] = _mean(passes)
    if "ragas" in frameworks:
        from ragas.metrics.collections import ContextRecall, Faithfulness

        llm = ragas_llm(judge)
        scores["faithfulness_ragas"] = _mean([_value(Faithfulness(llm=llm).score(
            user_input=case.question, response=response, retrieved_contexts=ctx)) for _ in range(judge_reps)])
        if case.reference:
            scores["context_recall_ragas"] = _mean([_value(ContextRecall(llm=llm).score(
                user_input=case.question, retrieved_contexts=ctx, reference=case.reference))
                for _ in range(judge_reps)])
    return scores, why


def _value(result) -> float | None:
    v = getattr(result, "value", result)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


class Runner:
    def __init__(self, system: RAGSystem, out_dir: Path, mode: str = "app", reps: int = 1,
                 judge_factory=ClaudeJudge, frameworks: set[str] | None = None, judge_reps: int = 1):
        self.judge_reps = judge_reps
        self.system, self.dir, self.mode, self.reps = system, Path(out_dir), mode, reps
        (self.dir / "traces").mkdir(parents=True, exist_ok=True)
        self.judge_factory = judge_factory
        wanted = {"deepeval", "ragas"} if frameworks is None else frameworks
        self.frameworks = {f for f in wanted if has(f)}
        missing = wanted - self.frameworks
        if missing:
            print(f"note: {', '.join(sorted(missing))} not installed - those metrics stay null", file=sys.stderr)
        self._lock = threading.Lock()

    def done(self) -> set[tuple[str, int]]:
        p = self.dir / "results.jsonl"
        if not p.exists():
            return set()
        return {(r["id"], r["rep"]) for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}

    def _append(self, name: str, row: dict) -> None:
        with self._lock, open(self.dir / name, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def run_case(self, case: Case, rep: int) -> None:
        t0 = time.perf_counter()
        retrieval = self.system.retrieve(case)
        answer = None
        if self.mode == "app":
            answer = self.system.generate(case, retrieval)
            response = answer.text
        elif self.mode == "oracle":
            response = case.reference
        elif self.mode == "copy":
            response = retrieval.contexts[0] if retrieval.contexts else ""
        else:
            response = ""
        judge = self.judge_factory()
        grade = programmatic(case, response)
        scores, why = judged(case, response, retrieval.contexts, judge, self.frameworks, self.judge_reps)
        grade.update(scores)
        status = "truncated" if answer and answer.stop_reason == "max_tokens" else "ok"
        (self.dir / "traces" / f"{case.id}_rep{rep}.json").write_text(json.dumps(
            [{"role": "user", "content": case.question},
             {"role": "context", "content": retrieval.contexts},
             {"role": "assistant", "content": response}], ensure_ascii=False, indent=1),
            encoding="utf-8", newline="\n")
        self._append("results.jsonl", {
            "id": case.id, "rep": rep, "group": case.group, "question": case.question, "mode": self.mode,
            "status": status, "grade": {k: grade.get(k) for k in METRICS}, "explanation": why,
            "model": answer.model if answer else None,
            "usage": {"input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens} if answer else None,
            "judge_model": judge.model, "judge_usage": judge.usage,
            "latency_s": round(time.perf_counter() - t0, 2), "retrieval_ms": round(retrieval.latency_ms, 1),
        })

    def run(self, cases: list[Case], workers: int = 4) -> None:
        done = self.done()
        todo = [(c, r) for c in cases for r in range(self.reps) if (c.id, r) not in done]
        print(f"{len(todo)} cases to run ({self.mode}) -> {self.dir}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.run_case, c, r): (c, r) for c, r in todo}
            for fut, (case, rep) in futures.items():
                try:
                    fut.result(timeout=CASE_TIMEOUT_S)
                    print(f"  ok  {case.id}#{rep}")
                except FutureTimeout:
                    self._error(case, rep, "timeout", f">{CASE_TIMEOUT_S}s")
                except JudgeError as exc:
                    self._error(case, rep, "judge_error", str(exc))
                except Exception as exc:  # noqa: BLE001 — harness errors are never model zeros
                    self._error(case, rep, "harness_error", f"{type(exc).__name__}: {exc}")

    def _error(self, case: Case, rep: int, kind: str, detail: str) -> None:
        print(f"  ERR {case.id}#{rep} {kind}: {detail[:200]}", file=sys.stderr)
        self._append("errors.jsonl", {"id": case.id, "rep": rep, "failure_class": kind, "detail": detail[:2000]})


def summarize(out_dir: Path) -> dict:
    """Means per metric over status=ok rows, with a rough 95 % interval."""
    rows = [json.loads(line) for line in (Path(out_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = [r for r in rows if r["status"] == "ok"]
    out: dict = {"n_rows": len(rows), "n_ok": len(ok)}
    for m in METRICS:
        vals = [r["grade"][m] for r in ok if r["grade"].get(m) is not None]
        if not vals:
            continue
        mu = mean(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        out[m] = {"mean": round(mu, 3), "ci95": round(1.96 * sd / len(vals) ** 0.5, 3), "n": len(vals)}
    errors = Path(out_dir) / "errors.jsonl"
    out["errors"] = len(errors.read_text(encoding="utf-8").splitlines()) if errors.exists() else 0
    out["judge_tokens"] = {k: sum(r["judge_usage"][k] for r in rows) for k in ("input_tokens", "output_tokens", "calls")}
    return out


def agreement(out_dir: Path) -> dict | None:
    """How well the two faithfulness implementations agree — your judge calibration signal."""
    rows = [json.loads(line) for line in (Path(out_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = [(r["grade"]["faithfulness_de"], r["grade"]["faithfulness_ragas"]) for r in rows
             if r["grade"].get("faithfulness_de") is not None and r["grade"].get("faithfulness_ragas") is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs, strict=True)
    mx, my = mean(xs), mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    r = sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy) if sx and sy else None
    diffs = [abs(x - y) for x, y in pairs]
    return {"n": len(pairs), "pearson_r": None if r is None else round(r, 3),
            "mean_abs_diff": round(mean(diffs), 3), "within_0_2": round(sum(d <= 0.2 for d in diffs) / len(diffs), 3)}


def regrade(out_dir: Path, cases: list[Case]) -> int:
    """Recompute programmatic metrics from stored traces — no API calls. Run for EVERY variant
    after changing a programmatic grader, or you compare scores from two different graders."""
    by_id = {c.id: c for c in cases}
    path = Path(out_dir) / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        trace = json.loads((Path(out_dir) / "traces" / f"{row['id']}_rep{row['rep']}.json").read_text(encoding="utf-8"))
        row["grade"].update(programmatic(by_id[row["id"]], trace[-1]["content"]))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return len(rows)


__all__ = ["Answer", "METRICS", "Runner", "agreement", "programmatic", "regrade", "summarize"]
