"""Offline tests: no API key, no DeepEval/RAGAS needed (framework metrics stay null)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.toy_rag.system import ToyRAG, make, make_truncating
from ragevalkit import answers, retrieval
from ragevalkit.cli import main as cli
from ragevalkit.dataset import Case, load_cases
from ragevalkit.judge import AbstentionVerdict, ClaudeJudge, JudgeError
from ragevalkit.text import fact_present, normalize

GOLDEN = Path(__file__).resolve().parents[1] / "examples" / "toy_rag" / "golden.yaml"
CASES = load_cases(GOLDEN)


# ── text & dataset ───────────────────────────────────────────────────────────

def test_normalize_formatting_variants():
    assert normalize("50 %  of\nfees") == "50% of fees"
    assert fact_present("450 EUR", "Partners bill 450 € per hour")
    assert not fact_present("450 EUR", "4500 EUR")
    assert fact_present(["data protection officer", "DPO"], "Tell the DPO first")


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("cases:\n  - {id: A, question: q}\n  - {id: A, question: r}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)


def test_echoing_a_forbidden_term_from_the_question_is_not_a_leak():
    c = Case(id="x", question="What about Acme?", forbidden_terms=["Acme", "gross negligence"])
    assert c.terms_that_must_not_appear() == ["gross negligence"]


# ── retrieval ────────────────────────────────────────────────────────────────

def test_toy_system_retrieval_is_clean():
    summary, _ = retrieval.evaluate(make(), CASES)
    assert summary["hit_at_k"] == 1.0 and summary["fact_recall"] == 1.0 and summary["leaks"] == 0


def test_truncated_context_is_caught_by_fact_recall_not_by_hit_rate():
    good, _ = retrieval.evaluate(make(), CASES)
    bad, _ = retrieval.evaluate(make_truncating(), CASES)
    assert bad["hit_at_k"] == good["hit_at_k"] == 1.0  # retrieval itself still perfect
    assert bad["fact_recall"] < good["fact_recall"]
    assert any(p.startswith("fact_recall") for p in retrieval.compare(bad, good))


class _FilterRemoved(ToyRAG):
    """Simulates a regression: the retriever no longer filters by tenant. Grading still uses the real ACL."""

    def retrieve(self, case):
        real_acl = self.can_see
        self.can_see = lambda c, h: True
        try:
            return super().retrieve(case)
        finally:
            self.can_see = real_acl


def test_acl_leak_is_always_a_regression():
    summary, _ = retrieval.evaluate(_FilterRemoved(), CASES)
    assert summary["leaks"] > 0
    assert retrieval.compare(summary, summary)[0].startswith("leaks")


def test_cli_retrieval_gate_exit_codes(tmp_path, capsys):
    base = tmp_path / "baseline.json"
    args = ["retrieval", "--system", "examples.toy_rag.system:make", "--cases", str(GOLDEN), "--baseline", str(base)]
    assert cli([*args, "--update-baseline"]) == 0
    assert cli(args) == 0
    assert cli(["retrieval", "--system", "examples.toy_rag.system:make_truncating", "--cases", str(GOLDEN),
                "--baseline", str(base)]) == 1
    assert "REGRESSION: fact_recall" in capsys.readouterr().err


# ── answers (fake LLM + fake judge) ──────────────────────────────────────────

class FakeLLM:
    """Answers from context; says it has no information when context does not cover the question."""

    def __init__(self, broken_ids=()):
        self.broken = set(broken_ids)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        prompt = kw["messages"][0]["content"]
        if any(f"Question: {q}" in prompt for q in self.broken):
            raise RuntimeError("simulated outage")
        ctx = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        text = ctx.split("\n---\n")[0]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], model="fake-model",
                               stop_reason="end_turn", usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class FakeJudge(ClaudeJudge):
    def __init__(self):
        super().__init__(model="fake-judge", client=object())

    def parse(self, prompt, schema):
        assert schema is AbstentionVerdict
        response = prompt.split("<response>", 1)[1].split("</response>", 1)[0]
        abstains = "no information" in response.lower() or not response.strip()
        return AbstentionVerdict(abstains=abstains, invents_specifics=not abstains and bool(response.strip()),
                                 reason="fake")


def run_mode(tmp_path, mode, system=None):
    out = tmp_path / mode
    answers.Runner(system or ToyRAG(client=FakeLLM()), out, mode=mode, judge_factory=FakeJudge,
                   frameworks=set()).run(CASES, workers=2)
    return out, answers.summarize(out)


def test_oracle_passes_programmatic_and_abstention(tmp_path):
    _, s = run_mode(tmp_path, "oracle")
    assert s["fact_coverage"]["mean"] == 1.0 and s["abstention"]["mean"] == 1.0 and s["errors"] == 0


def test_null_fails_everything_it_can(tmp_path):
    out, s = run_mode(tmp_path, "null")
    assert s["fact_coverage"]["mean"] == 0.0
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    # empty answer: faithfulness is NOT scored (would be a vacuous 1.0), relevancy is 0
    assert all(r["grade"]["faithfulness_de"] is None for r in rows)


def test_app_mode_detects_leaked_terms(tmp_path):
    out, s = run_mode(tmp_path, "app", _FilterRemoved(client=FakeLLM()))  # ben can now read Acme
    rows = {r["id"]: r for r in map(json.loads, (out / "results.jsonl").read_text(encoding="utf-8").splitlines())}
    assert rows["X01"]["grade"]["terms_safe"] == 0.0      # "gross negligence" from Acme's contract
    assert rows["X01"]["grade"]["abstention"] == 0.0


def test_harness_errors_go_to_sidecar_not_results(tmp_path):
    broken = CASES[0].question
    out, s = run_mode(tmp_path, "app", ToyRAG(client=FakeLLM(broken_ids=[broken])))
    errors = [json.loads(line) for line in (out / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["id"] for e in errors] == [CASES[0].id] and s["n_rows"] == len(CASES) - 1


def test_resume_and_regrade(tmp_path):
    out, _ = run_mode(tmp_path, "oracle")
    answers.Runner(ToyRAG(client=FakeLLM()), out, mode="oracle", judge_factory=FakeJudge,
                   frameworks=set()).run(CASES)  # nothing left to do
    assert answers.summarize(out)["n_rows"] == len(CASES)
    assert answers.regrade(out, CASES) == len(CASES)


def test_judge_rejects_wrong_served_model():
    class Client:
        class messages:  # noqa: N801
            @staticmethod
            def parse(**kw):
                return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn",
                                       parsed_output=AbstentionVerdict(abstains=True, invents_specifics=False,
                                                                       reason="x"), model="some-other-model")

    with pytest.raises(JudgeError, match="served by"):
        ClaudeJudge(model="claude-sonnet-5-5", client=Client()).parse("p", AbstentionVerdict)
