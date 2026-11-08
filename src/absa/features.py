"""
Hand-crafted feature extractors used next to TF-IDF.

Two scikit-learn compatible transformers:

  WindowLexiconFeatures  - for the ASPECT SENTIMENT model. Input: list of window
                           strings. Output: small non-negative count matrix
                           (so it also works with Naive Bayes).

  IncongruityFeatures    - for the SARCASM model. Input: DataFrame/records with
                           'text' (raw review) and optional 'rating' (1-5 stars).
                           Output: the sentiment-incongruity feature vector.

SENTIMENT-INCONGRUITY (advanced concept #2) - the idea in one line:
    Sarcasm = saying something positive while meaning something negative.
    So the TEXT ITSELF usually contains a clash: praise words next to a bad
    situation ("wah, 2 ghante late, great service"), a happy emoji next to a
    complaint, or 5 praise words on a 1-star review. We turn each kind of clash
    into a number the classifier can weigh:

      * pos_count / neg_count   - lexicon hits (both high = incongruent)
      * incongruity_flag        - 1 if BOTH positive and negative words appear
      * incongruity_strength    - strongest positive x strongest negative
      * marker_count            - mock-praise markers ("wah", "great job", clap emoji)
      * marker_with_neg         - marker present AND negative word present
      * emoji_text_clash        - emoji polarity disagrees with word polarity
      * rating_text_clash       - 1-2 star rating but positive words dominate
                                  (or 4-5 star but negative words dominate)
      * elongation, '!', '?', ALL-CAPS counts - exaggeration cues
      * negation count, length, code-mixing index

    This is the classic "incongruity" family of features from the sarcasm
    literature (Riloff et al. 2013; Joshi et al. 2015 for English), which Shah &
    Maurya (ICON 2021) argue is under-used for code-mixed text. We implement it
    with transparent counts instead of a neural self-matching network, which
    keeps it CPU-cheap and fully explainable.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import lexicon as LX
from .normalize import normalize

SENT_FEATURE_NAMES = ["win_pos_count", "win_neg_count", "win_negator_count",
                      "win_intensifier_count", "win_pos_max", "win_neg_max",
                      "win_len"]

SARC_FEATURE_NAMES = ["pos_count", "neg_count", "incongruity_flag", "incongruity_strength",
                      "marker_count", "marker_with_neg", "emoji_pos", "emoji_neg",
                      "emoji_text_clash", "rating_scaled", "rating_missing",
                      "rating_text_clash", "n_elongations", "n_exclaim", "n_question",
                      "n_caps_words", "negator_count", "intensifier_count",
                      "log_len", "cmi"]


class WindowLexiconFeatures(BaseEstimator, TransformerMixin):
    """Counts of lexicon words inside an aspect window (non-negative)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[str]):
        rows = []
        for win in X:
            toks = win.split()
            pos = [LX.POSITIVE[t] for t in toks if t in LX.POSITIVE]
            neg = [-LX.NEGATIVE[t] for t in toks if t in LX.NEGATIVE]
            rows.append([
                len(pos), len(neg),
                sum(t in LX.NEGATORS for t in toks),
                sum(t in LX.INTENSIFIERS for t in toks),
                max(pos) if pos else 0.0,
                max(neg) if neg else 0.0,
                min(len(toks), 30) / 30.0,
            ])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(SENT_FEATURE_NAMES, dtype=object)


def _as_records(X) -> List[dict]:
    if isinstance(X, pd.DataFrame):
        return X.to_dict("records")
    out = []
    for x in X:
        if isinstance(x, str):
            out.append({"text": x, "rating": np.nan})
        else:
            out.append(dict(x))
    return out


class IncongruityFeatures(BaseEstimator, TransformerMixin):
    """Sentiment-incongruity feature vector for sarcasm detection."""

    def __init__(self, lid=None):
        self.lid = lid   # optional LanguageIdentifier, used for the CMI feature

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _features_for(rec: dict, lid=None) -> List[float]:
        norm = normalize(rec.get("text", ""))
        toks = norm.tokens
        pos = [LX.POSITIVE[t] for t in toks if t in LX.POSITIVE]
        neg = [-LX.NEGATIVE[t] for t in toks if t in LX.NEGATIVE]
        pos_max = max(pos) if pos else 0.0
        neg_max = max(neg) if neg else 0.0
        markers = sum(t in LX.SARCASM_MARKERS for t in toks)
        emoji_toks = [t for t in toks if t.startswith("emoji_")]
        emoji_pos = sum(LX.POSITIVE.get(t, 0) for t in emoji_toks)
        emoji_neg = -sum(LX.NEGATIVE.get(t, 0) for t in emoji_toks)
        word_pol = sum(pos) - sum(neg)
        emoji_pol = emoji_pos - emoji_neg
        emoji_clash = 1.0 if (emoji_pol > 0 and word_pol < 0) or (emoji_pol < 0 and word_pol > 0) else 0.0

        rating = rec.get("rating", np.nan)
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            rating = np.nan
        rating_missing = 1.0 if np.isnan(rating) else 0.0
        rating_scaled = 0.0 if rating_missing else (rating - 3.0) / 2.0
        rating_clash = 0.0
        if not rating_missing:
            if rating <= 2 and sum(pos) > sum(neg):
                rating_clash = 1.0
            elif rating >= 4 and sum(neg) > sum(pos):
                rating_clash = 1.0

        cmi_val = 0.0
        if lid is not None:
            try:
                cmi_val = lid.analyse(toks).cmi / 50.0
            except Exception:
                cmi_val = 0.0

        return [
            len(pos), len(neg),
            1.0 if (pos and neg) else 0.0,
            pos_max * neg_max,
            markers,
            1.0 if (markers and neg) else 0.0,
            emoji_pos, emoji_neg, emoji_clash,
            rating_scaled, rating_missing, rating_clash,
            norm.n_elongations, norm.n_exclaim, norm.n_question, norm.n_caps_words,
            sum(t in LX.NEGATORS for t in toks),
            sum(t in LX.INTENSIFIERS for t in toks),
            float(np.log1p(len(toks))),
            cmi_val,
        ]

    def transform(self, X):
        recs = _as_records(X)
        return np.asarray([self._features_for(r, self.lid) for r in recs], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(SARC_FEATURE_NAMES, dtype=object)


def incongruity_report(text: str, rating: Optional[float] = None, lid=None) -> dict:
    """Human-readable version of the incongruity features (used by the demo app)."""
    vals = IncongruityFeatures._features_for({"text": text, "rating": rating}, lid)
    return dict(zip(SARC_FEATURE_NAMES, vals))
