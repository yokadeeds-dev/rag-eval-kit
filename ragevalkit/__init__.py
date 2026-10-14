"""rag-eval-kit — evaluate a RAG system through its real code path.

Deterministic retrieval metrics (CI gate), answer metrics with DeepEval + RAGAS via a Claude judge
that works with current models, grader self-checks (oracle / null / copy), resumable runs.
"""

from ragevalkit.dataset import Case, load_cases
from ragevalkit.system import Answer, Hit, RAGSystem, Retrieval

__all__ = ["Answer", "Case", "Hit", "RAGSystem", "Retrieval", "load_cases"]
__version__ = "0.1.0"
