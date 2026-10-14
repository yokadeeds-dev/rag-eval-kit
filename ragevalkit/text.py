"""Text normalisation for programmatic graders.

Exact substring checks reject correct answers for formatting reasons ("50 %" vs "50%",
"450 €" vs "450 EUR", a reworded noun). Two such false negatives cost a full re-grade in
the case study — normalise both sides and allow equivalent phrasings per fact.
"""

from __future__ import annotations

import re

_CURRENCY = [("€", " eur"), ("euro", "eur"), ("£", " gbp"), ("$", " usd")]


def normalize(text: str) -> str:
    t = text.lower()
    for a, b in _CURRENCY:
        t = t.replace(a, b)
    t = t.replace("‑", "-").replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+%", "%", t)
    return t.strip()


def fact_present(fact: str | list[str], text: str) -> bool:
    """A fact is a string or a list of equivalent phrasings; any one counts."""
    alternatives = fact if isinstance(fact, list) else [fact]
    norm = normalize(text)
    return any(normalize(a) in norm for a in alternatives)


def fact_share(facts: list, text: str) -> float | None:
    if not facts:
        return None
    return sum(fact_present(f, text) for f in facts) / len(facts)
