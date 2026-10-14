"""A deliberately tiny RAG system to show the adapter — and the bug class fact_recall catches.

Two tenants (acme, globex) plus public policies. Lexical scoring, no model download.
`context_chars` simulates the classic truncation bug: the right document is retrieved, but only
its first N characters reach the LLM.

    rag-eval retrieval --system examples.toy_rag.system:make_truncating --cases examples/toy_rag/golden.yaml
    rag-eval retrieval --system examples.toy_rag.system:make --cases examples/toy_rag/golden.yaml
"""

from __future__ import annotations

import os
import re
import time

from ragevalkit import Answer, Case, Hit, Retrieval

DOCS = [
    Hit("pol-1", "Incident protocol: data breach", metadata={"tenant": None, "text":
        "Incident protocol: data breach. Suspected breaches must be reported to the data protection "
        "officer within 2 hours. The supervisory authority must be notified within 72 hours."}),
    Hit("pol-2", "Travel and expenses policy", metadata={"tenant": None, "text":
        "Travel and expenses policy. Economy class for flights under six hours. Hotel bookings go through "
        "the travel desk. Mileage is reimbursed at 0.30 EUR per kilometre."}),
    Hit("pol-3", "Remote work policy", metadata={"tenant": None, "text":
        "Remote work policy. Staff may work remotely up to three days a week. Client data may only be "
        "processed on company devices; private devices are prohibited."}),
    Hit("acme-1", "Acme service agreement", metadata={"tenant": "acme", "text":
        "Acme service agreement. Payment is due within 30 days of invoice. Liability is limited to "
        "intent and gross negligence. Either party may terminate with 8 weeks notice."}),
    Hit("acme-2", "Acme licence agreement", metadata={"tenant": "acme", "text":
        "Acme licence agreement. The licence covers 50 named users. Venue for disputes is Hamburg."}),
    Hit("glx-1", "Globex purchase agreement", metadata={"tenant": "globex", "text":
        "Globex purchase agreement. Payment is due within 14 days. Liability is capped at the contract value."}),
]
USERS = {"anna": {"acme"}, "ben": {"globex"}}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class ToyRAG:
    def __init__(self, context_chars: int | None = None, client=None, model: str | None = None):
        self.context_chars = context_chars
        self.client = client
        self.model = model or os.getenv("TOY_RAG_MODEL", "claude-opus-5-5")

    def can_see(self, case: Case, hit: Hit) -> bool:
        tenant = hit.metadata.get("tenant")
        return tenant is None or tenant in USERS.get(case.user, set())

    def corpus(self) -> list[Hit]:
        return DOCS

    def retrieve(self, case: Case) -> Retrieval:
        t0 = time.perf_counter()
        q = _tokens(case.question)
        visible = [d for d in DOCS if self.can_see(case, d)]  # ACL filter BEFORE ranking
        ranked = sorted(visible, key=lambda d: len(q & _tokens(d.metadata["text"])), reverse=True)
        hits = [Hit(d.id, d.title, float(len(q & _tokens(d.metadata["text"]))), d.metadata) for d in ranked[:3]]
        texts = [h.metadata["text"] for h in hits]
        if self.context_chars:
            texts = [t[: self.context_chars] for t in texts]
        return Retrieval(hits, texts, (time.perf_counter() - t0) * 1000)

    def generate(self, case: Case, retrieval: Retrieval) -> Answer:
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        context = "\n---\n".join(retrieval.contexts) or "(no information available)"
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024,
            system=("Answer only from the context. If the context does not cover the question or concerns a "
                    "different company, say that you have no information."),
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {case.question}"}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "text") == "text")
        return Answer(text, model=getattr(msg, "model", ""), stop_reason=getattr(msg, "stop_reason", ""),
                      input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                      latency_ms=(time.perf_counter() - t0) * 1000)


def make() -> ToyRAG:
    return ToyRAG()


def make_truncating() -> ToyRAG:
    return ToyRAG(context_chars=80)
