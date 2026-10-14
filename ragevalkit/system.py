"""The adapter contract: how rag-eval-kit talks to YOUR RAG system.

Implement `RAGSystem` (or anything with the same methods) and point the CLI at a factory:

    rag-eval retrieval --system mypackage.eval_adapter:make_system --cases golden.yaml

The adapter must call your real code path — the same retrieval, prompt assembly and LLM
call your endpoint uses. An eval that re-implements the pipeline measures something else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ragevalkit.dataset import Case


@dataclass
class Hit:
    """One retrieved document/chunk."""
    id: str
    title: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Retrieval:
    hits: list[Hit]              # everything the retriever returned, in rank order
    contexts: list[str]          # exactly the text the LLM will see as context
    latency_ms: float = 0.0


@dataclass
class Answer:
    text: str
    model: str = ""              # model that served the request, read from the API response
    stop_reason: str = ""        # "max_tokens" marks the row as truncated, not wrong
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@runtime_checkable
class RAGSystem(Protocol):
    def retrieve(self, case: Case) -> Retrieval:
        """Run retrieval for case.question as case.user (ACL / tenant) would."""

    def generate(self, case: Case, retrieval: Retrieval) -> Answer:
        """Produce the answer from exactly these contexts, through your real prompt + LLM call."""

    def corpus(self) -> list[Hit]:
        """All documents (id/title/metadata), used to count how many relevant ones exist.

        Return [] if listing the corpus is not possible; precision@k then falls back to k.
        """

    def can_see(self, case: Case, hit: Hit) -> bool:
        """Whether case.user may see this hit. Return True if your system has no ACL."""
