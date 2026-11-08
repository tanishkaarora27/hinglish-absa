"""
Stage 3 - Aspect extraction and Stage 4a - aspect-conditioned context windows.

ASPECT EXTRACTION (seed lexicon + rules)
    A review like "delivery fast thi but doodh expired tha" talks about two
    aspects. We detect them by matching normalised tokens against the seed
    lexicon in lexicon.ASPECT_LEXICON. Rules on top of the lexicon:
      * multi-token phrases are matched first (longest match wins);
      * each matched token is a "mention" with a position (token index);
      * a review with no mention is assigned the aspect "general" so overall
        sentiment is still produced (common for very short reviews like
        "bakwas app" -> app_support, or "bahut accha" -> general).

ASPECT-CONDITIONED CONTEXT WINDOW (advanced concept #1)
    Problem: a whole-review classifier gives ONE sentiment for the whole text,
    so "delivery fast thi but doodh expired tha" would be labelled... what?
    The sentiment that matters for "delivery_speed" lives NEAR the word
    "delivery"; the sentiment for "product_quality" lives near "doodh".
    Solution: for each aspect mention we cut out the tokens within +-K
    positions of the mention, and we STOP the window at clause breakers
    ("but", "lekin", "par", ".", "!"). The sentiment classifier then sees only
    "delivery fast thi" for delivery_speed and "doodh expired tha" for
    product_quality. This is the simplest form of "aspect-conditioned"
    classification and is exactly the +-window heuristic used in classic ABSA
    work before transformers.

    K is a hyper-parameter (default 5). The notebook has an ablation that
    compares K=3,5,8 and "no window" (full text).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .lexicon import ASPECT_LEXICON, CLAUSE_BREAKERS

ASPECTS = list(ASPECT_LEXICON.keys()) + ["general"]

# Build a flat lookup: token -> aspect, and phrase list for multi-token triggers.
_SINGLE: Dict[str, str] = {}
_PHRASES: List[tuple[tuple[str, ...], str]] = []
for _aspect, _triggers in ASPECT_LEXICON.items():
    for _t in _triggers:
        parts = tuple(_t.split())
        if len(parts) == 1:
            _SINGLE.setdefault(parts[0], _aspect)
        else:
            _PHRASES.append((parts, _aspect))
_PHRASES.sort(key=lambda p: -len(p[0]))   # longest phrase first


@dataclass
class Mention:
    aspect: str
    trigger: str        # the token/phrase that fired
    start: int          # token index (inclusive)
    end: int            # token index (exclusive)


def extract_mentions(tokens: Sequence[str]) -> List[Mention]:
    """Find every aspect trigger in a normalised token list."""
    mentions: List[Mention] = []
    i, n = 0, len(tokens)
    while i < n:
        matched = False
        for parts, aspect in _PHRASES:
            L = len(parts)
            if tuple(tokens[i:i + L]) == parts:
                mentions.append(Mention(aspect, " ".join(parts), i, i + L))
                i += L
                matched = True
                break
        if matched:
            continue
        tok = tokens[i]
        if tok in _SINGLE:
            mentions.append(Mention(_SINGLE[tok], tok, i, i + 1))
        i += 1
    return mentions


def detect_aspects(tokens: Sequence[str]) -> List[str]:
    """Unique aspects in order of first appearance; 'general' if none."""
    seen: List[str] = []
    for m in extract_mentions(tokens):
        if m.aspect not in seen:
            seen.append(m.aspect)
    return seen or ["general"]


def window_range(tokens: Sequence[str], start: int, end: int, k: int = 5,
                 stop_at_clause: bool = True, boundaries: set | None = None) -> tuple[int, int]:
    """
    Index range [lo, hi) of the window around mention [start, end):
      * at most k tokens on each side,
      * cut at clause breakers ("but", "lekin", "!", ...) when stop_at_clause,
      * cut at positions in `boundaries` (tokens that trigger a DIFFERENT aspect),
        so "delivery fast thi doodh expired" gives delivery_speed -> "delivery fast thi".
    """
    boundaries = boundaries or set()
    lo = max(0, start - k)
    hi = min(len(tokens), end + k)
    for j in range(start - 1, lo - 1, -1):
        if (stop_at_clause and tokens[j] in CLAUSE_BREAKERS) or j in boundaries:
            lo = j + 1
            break
    for j in range(end, hi):
        if (stop_at_clause and tokens[j] in CLAUSE_BREAKERS) or j in boundaries:
            hi = j
            break
    return lo, hi


def context_window(tokens: Sequence[str], start: int, end: int, k: int = 5,
                   stop_at_clause: bool = True, boundaries: set | None = None) -> List[str]:
    lo, hi = window_range(tokens, start, end, k, stop_at_clause, boundaries)
    return list(tokens[lo:hi])


def aspect_windows(tokens: Sequence[str], k: int = 5, stop_at_clause: bool = True,
                   use_window: bool = True) -> Dict[str, str]:
    """
    Map aspect -> window text. If an aspect is mentioned several times, the
    windows are MERGED (union of covered positions, kept in sentence order) so
    nothing is duplicated. With use_window=False the full text is returned for
    every detected aspect (this is the 'no window' ablation setting).
    """
    mentions = extract_mentions(tokens)
    if not mentions:
        return {"general": " ".join(tokens)}
    if not use_window:
        full = " ".join(tokens)
        return {a: full for a in dict.fromkeys(m.aspect for m in mentions)}
    covered: Dict[str, set] = {}
    for m in mentions:
        others = {i for o in mentions if o.aspect != m.aspect for i in range(o.start, o.end)}
        lo, hi = window_range(tokens, m.start, m.end, k, stop_at_clause, others)
        covered.setdefault(m.aspect, set()).update(range(lo, hi))
    return {a: " ".join(tokens[i] for i in sorted(idx)) for a, idx in covered.items()}
