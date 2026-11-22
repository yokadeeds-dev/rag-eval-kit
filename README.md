# rag-eval-kit

Evaluate a RAG system **through its real code path**: a free, deterministic retrieval gate for CI, and answer metrics with **DeepEval + RAGAS** scored by a Claude judge that works with current models — plus the grader self-checks that keep the numbers honest.

Extracted from a real evaluation of a legal RAG assistant, where it found that the retriever was perfect but the LLM never saw 22 % of the required facts ([case study](https://github.com/yokadeeds-dev/EVL/blob/main/03_Prototyp/backend/eval/RESULTS.md)).

## Case study: what it found in EVL

25 golden questions on a multi-tenant legal assistant (FastAPI, pgvector, Claude), synthetic corpus, total evaluation cost ≈ $5.75.

| Finding | Fix | Effect |
|---|---|---|
| Context cut after 200 characters — right documents retrieved, 22 % of required facts never reached the LLM | pass the whole document | facts in answer 0.91 → **1.00**, faithfulness (RAGAS) 0.34 → **0.61** |
| More context made the model answer a question about a *blocked* client with the user's *own* client's contracts (wrong answer, no leak) | shared grounding rule in all prompts | abstention on blocked knowledge 0.40 → **1.00** |
| Phone numbers like `+49 170 …` slipped past the PII filter | regex fix, found by the red-team suite | 9/9 red-team checks pass |

Open by design: ~40 % of statements are not directly backed by the context — a template asks for more than the sources hold; a product decision, measured but not changed. Limits (25 cases, synthetic data, judge from the same model family) are listed in the [full results](https://github.com/yokadeeds-dev/EVL/blob/main/03_Prototyp/backend/eval/RESULTS.md).

## Why another eval kit

Four things that went wrong with off-the-shelf tooling, and what this kit does instead:

| Problem found in practice | What the kit does |
|---|---|
| DeepEval's and RAGAS' built-in Anthropic integrations send forced `tool_choice` / `temperature` → **HTTP 400** on Claude Opus 5.5 / Sonnet 5.5 | One judge over **structured outputs** (`messages.parse`), wired into both frameworks |
| DeepEval faithfulness (default) counts only *contradictions* as unfaithful — a claim missing from the context scored **1.0** | `penalize_ambiguous_claims=True`; RAGAS runs alongside, and their **agreement** is reported |
| RAGAS rewrote "you must…" into "*the data controller* must…" and then rejected it — the same answer scored **0.0 and 1.0** on two passes | `--judge-reps N` averages passes; toy example: agreement r **0.09 → 0.68** with 3 passes |
| Exact-match graders rejected correct answers ("50 %" ≠ "50%") | Normalisation + equivalent phrasings per fact; `regrade` re-scores stored runs for free |

## Install

```bash
pip install -e .                    # core: retrieval gate, programmatic graders, Claude judge
pip install -e ".[deepeval,ragas]"  # answer metrics from both frameworks
```

## Quickstart (toy example, no API key)

```bash
rag-eval retrieval --system examples.toy_rag.system:make --cases examples/toy_rag/golden.yaml
rag-eval retrieval --system examples.toy_rag.system:make_truncating --cases examples/toy_rag/golden.yaml
```

The second system truncates contexts to 80 characters. Retrieval metrics stay perfect — only **facts in context** drops (1.00 → 0.33). That is the bug class this kit was built to catch.

## Plug in your system

```python
# my_project/eval_adapter.py
from ragevalkit import Answer, Hit, Retrieval

class MyRAG:
    def retrieve(self, case) -> Retrieval:
        results = my_search(case.question, user=case.user)          # your real retriever + ACL
        return Retrieval(hits=[Hit(r.id, r.title, r.score, r.meta) for r in results],
                         contexts=build_context(results))           # exactly what the LLM sees

    def generate(self, case, retrieval) -> Answer:
        msg = my_llm_call(case.question, retrieval.contexts)        # your real prompt + model
        return Answer(msg.text, model=msg.model, stop_reason=msg.stop_reason,
                      input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)

    def corpus(self):            return [Hit(d.id, d.title, metadata=d.meta) for d in all_docs()]
    def can_see(self, case, hit): return acl_allows(case.user, hit)

def make_system():
    return MyRAG()
```

```bash
rag-eval retrieval --system my_project.eval_adapter:make_system --cases golden.yaml \
                   --baseline eval/baseline.json --update-baseline      # once
rag-eval retrieval --system my_project.eval_adapter:make_system --cases golden.yaml \
                   --baseline eval/baseline.json                        # CI gate: exit 1 on regression or leak
```

The adapter must call your **real** pipeline. Re-implementing retrieval or the prompt in the eval measures a different system.

## Golden set

```yaml
cases:
  - id: L01
    group: policies
    user: alice                                  # passed to your adapter (ACL / tenant)
    question: How fast must a data breach be reported?
    relevant: {title_prefix: ["Incident protocol"]}      # or ids: [...], metadata: {k: v}; {} = abstain case
    facts: ["72 hours", ["DPO", "data protection officer"]]
    reference: The authority must be notified within 72 hours.
    forbidden_terms: ["Contoso"]                 # must not appear in the answer
    forbidden_metadata: {tenant: [T2]}           # hits with these values are leaks
```

Include both directions: answerable cases, **abstain** cases (`relevant: {}`) and **ACL** cases where relevant documents exist but belong to someone else.

## Metrics

| Retrieval (free) | |
|---|---|
| `hit_at_k`, `mrr` | rank of the first relevant hit |
| `precision_at_k` | normalised by the number of relevant docs in the corpus (1/1 is reachable) |
| `fact_recall` | share of facts present in the **LLM context** — catches truncation and chunking bugs |
| `leaks` | hits the user must not see — any leak fails the gate |

| Answers (LLM + judge) | |
|---|---|
| `fact_coverage`, `terms_safe` | programmatic |
| `faithfulness_de`, `relevancy_de` | DeepEval |
| `faithfulness_ragas`, `context_recall_ragas` | RAGAS |
| `abstention` | Claude judge: abstains instead of inventing — including facts about a *different* entity |

```bash
rag-eval answers --system ... --cases golden.yaml --out runs/baseline           # ~$0.03 / case (Sonnet 5.5 judge)
rag-eval answers --system ... --cases golden.yaml --out runs/baseline --judge-reps 3
rag-eval summary --out runs/baseline
```

Runs write `results.jsonl` (one row per case, as it completes), `traces/`, and `errors.jsonl`. API and harness failures go to the error file — never scored as a model zero. Runs resume per `(case, rep)`. Truncated answers are flagged, not averaged in.

## Check the graders before the model

```bash
rag-eval answers ... --out runs/check_oracle --mode oracle   # reference answers: should score high
rag-eval answers ... --out runs/check_null   --mode null     # empty answers: should score ~0
rag-eval answers ... --out runs/check_copy   --mode copy     # verbatim context: faithfulness ~1
```

If the oracle fails, the grader is broken; if the null passes, the grader is too lenient. After editing a programmatic grader, run `rag-eval regrade` for **every** variant before comparing them.

## Limits

- Judge metrics are only as good as the judge. Report `agreement`, read the cases where the two frameworks disagree, and size the set against the effect you want to see (25 cases × 1 rep ≈ ±0.1 on faithfulness).
- Judge and system from the same model family can share blind spots.
- The judge has no refusal fallback on purpose: a silent model switch would make scores incomparable.

## Development

```bash
pip install -e ".[dev]" && pytest && ruff check .     # offline, no API key
```

MIT licensed.
