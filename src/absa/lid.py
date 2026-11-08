"""
Stage 2 - Word-level Language Identification (LID) and Code-Mixing Index (CMI).

Why this matters (viva answer):
    "Delivery bahut late thi" mixes English (delivery, late) and Hindi (bahut,
    thi) in ONE sentence. Knowing which word is which language lets us
    (a) measure how code-mixed each review is (CMI), which we report as a dataset
    statistic and use as a feature, and (b) apply the right lexicon to the
    right word. We do NOT need a heavy model for this: a character n-gram
    logistic-regression classifier trained on a public word-level LID corpus
    (L3Cube-HingLID, CC BY-NC-SA 4.0) reaches high accuracy on CPU in seconds.

How the LID model works:
    Each word is turned into character n-grams (2-4 chars, with word boundary
    markers). "bahut" -> "<b", "ba", "ah", "hu", "ut", "t>", "<ba", ... Hindi
    words written in Roman script have characteristic letter patterns
    ("aa", "hai", "kar", "..ta hai") that English words rarely have. A linear
    classifier over these n-grams learns those patterns.

Code-Mixing Index (Das & Gambäck, 2014):
    CMI = 100 * (1 - max_lang_tokens / total_lang_tokens)  (for tokens that got
    a language tag, ignoring "other" such as numbers / emoji).
    CMI = 0 means monolingual; the more balanced the mix, the higher the CMI
    (max 50 for two languages). We compute it per review.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LANG_EN, LANG_HI, LANG_OTHER = "en", "hi", "other"

# Tokens that should never go through the classifier.
_OTHER_RE = re.compile(r"^(<num>|<url>|<user>|emoji_.*|!|\?)$")

# A tiny override dictionary for very frequent, very short tokens that character
# n-grams can get wrong. Everything else is decided by the trained model.
_OVERRIDE = {
    "hai": LANG_HI, "ka": LANG_HI, "ki": LANG_HI, "ke": LANG_HI, "ko": LANG_HI,
    "se": LANG_HI, "me": LANG_HI, "mein": LANG_HI, "par": LANG_HI, "aur": LANG_HI,
    "ya": LANG_HI, "toh": LANG_HI, "to": LANG_HI, "hi": LANG_HI, "bhi": LANG_HI,
    "nahi": LANG_HI, "na": LANG_HI, "mat": LANG_HI, "tha": LANG_HI, "thi": LANG_HI,
    "the": LANG_EN, "a": LANG_EN, "an": LANG_EN, "is": LANG_EN, "was": LANG_EN,
    "are": LANG_EN, "and": LANG_EN, "or": LANG_EN, "of": LANG_EN, "in": LANG_EN,
    "on": LANG_EN, "for": LANG_EN, "app": LANG_EN, "order": LANG_EN,
    "delivery": LANG_EN, "delivery_partner": LANG_EN, "customer_support": LANG_EN,
    "on_time": LANG_EN, "ten_minute": LANG_EN, "paisa_vasool": LANG_HI,
    "kya_baat": LANG_HI, "wah_bhai": LANG_HI,
}


def build_lid_model() -> Pipeline:
    """Character n-gram TF-IDF + logistic regression. Fast and CPU friendly."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])


def read_l3cube_lid(path: str, max_tokens: int | None = None) -> Tuple[List[str], List[str]]:
    """Parse the L3Cube-HingLID format: 'token<TAB>HI|EN' per line, blank line between tweets."""
    words, labels = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "\t" not in line:
                continue
            tok, lab = line.split("\t", 1)
            tok, lab = tok.strip().lower(), lab.strip().upper()
            if not re.fullmatch(r"[a-z']+", tok):
                continue  # skip numbers/punctuation; they are "other" by rule
            if lab not in ("HI", "EN"):
                continue
            words.append(tok)
            labels.append(LANG_HI if lab == "HI" else LANG_EN)
            if max_tokens and len(words) >= max_tokens:
                break
    return words, labels


def read_swami_lid(path: str) -> Tuple[List[str], List[str]]:
    """Parse Swami et al. (2018) 'Sarcasm_tweets_with_language.txt': 'token  lang' lines."""
    words, labels = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            tok, lab = parts[0].lower(), parts[1].lower()
            if lab in ("en", "hi") and re.fullmatch(r"[a-z']+", tok):
                words.append(tok)
                labels.append(lab)
    return words, labels


@dataclass
class LIDResult:
    tokens: List[str]
    langs: List[str]
    cmi: float
    frac_hi: float
    frac_en: float


class LanguageIdentifier:
    """Wraps the trained model with rule overrides and CMI computation."""

    def __init__(self, model: Pipeline | None = None):
        self.model = model

    def fit(self, words: Sequence[str], labels: Sequence[str]) -> "LanguageIdentifier":
        self.model = build_lid_model().fit(list(words), list(labels))
        return self

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LanguageIdentifier":
        return cls(joblib.load(path))

    def tag(self, tokens: Sequence[str]) -> List[str]:
        langs = [None] * len(tokens)
        to_predict, idx = [], []
        for i, tok in enumerate(tokens):
            if _OTHER_RE.match(tok):
                langs[i] = LANG_OTHER
            elif tok in _OVERRIDE:
                langs[i] = _OVERRIDE[tok]
            elif "_" in tok:            # phrase tokens default to English heads
                langs[i] = LANG_EN
            else:
                to_predict.append(tok)
                idx.append(i)
        if to_predict:
            if self.model is None:
                preds = [LANG_HI] * len(to_predict)   # conservative fallback
            else:
                preds = self.model.predict(to_predict)
            for i, p in zip(idx, preds):
                langs[i] = p
        return langs

    def analyse(self, tokens: Sequence[str]) -> LIDResult:
        langs = self.tag(tokens)
        counts = Counter(l for l in langs if l in (LANG_EN, LANG_HI))
        n = sum(counts.values())
        if n == 0:
            cmi = 0.0
        else:
            cmi = 100.0 * (1 - max(counts.values()) / n)
        return LIDResult(tokens=list(tokens), langs=langs, cmi=round(cmi, 2),
                         frac_hi=counts[LANG_HI] / n if n else 0.0,
                         frac_en=counts[LANG_EN] / n if n else 0.0)


def cmi(langs: Iterable[str]) -> float:
    """Code-Mixing Index for a list of language tags."""
    counts = Counter(l for l in langs if l in (LANG_EN, LANG_HI))
    n = sum(counts.values())
    return 0.0 if n == 0 else round(100.0 * (1 - max(counts.values()) / n), 2)
