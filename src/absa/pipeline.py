"""
End-to-end pipeline: normalise -> LID/CMI -> aspects -> windows -> aspect
sentiment -> sarcasm -> polarity correction.

Model choices (all CPU-only, scikit-learn):
    "nb"  : ComplementNB            (good with small data + imbalance)
    "lr"  : LogisticRegression      (class_weight='balanced', interpretable weights)
    "svm" : LinearSVC inside CalibratedClassifierCV (strong text baseline; the
            calibration wrapper gives probabilities needed by LIME / thresholds)

Features for both models: character n-gram TF-IDF (2-5, word-boundary aware)
+ word uni/bi-gram TF-IDF + hand-crafted lexicon features. Character n-grams
are THE key choice for Hinglish: "bakwas"/"bakwaas"/"bkwas" share most 3-grams,
so spelling noise that survives normalisation is still handled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from .aspects import ASPECTS, aspect_windows, detect_aspects
from .features import IncongruityFeatures, WindowLexiconFeatures, _as_records
from .lid import LanguageIdentifier
from .normalize import normalize

SENTIMENTS = ["negative", "neutral", "positive"]


# ----------------------------------------------------------------------------
# model builders
# ----------------------------------------------------------------------------
def _classifier(kind: str):
    if kind == "nb":
        return ComplementNB(alpha=0.3)
    if kind == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=5000, C=2.0)
    if kind == "svm":
        return CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.5), cv=3)
    raise ValueError(f"unknown model kind {kind!r}; use nb | lr | svm")


def _text_features(min_df: int = 1) -> FeatureUnion:
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df,
                                 sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df,
                                 token_pattern=r"\S+", sublinear_tf=True)),
    ])


def build_sentiment_model(kind: str = "lr", use_lexicon_feats: bool = True) -> Pipeline:
    """Input: list of window strings (already normalised). Output: sentiment label."""
    parts = [("text", _text_features())]
    if use_lexicon_feats:
        parts.append(("lex", WindowLexiconFeatures()))
    return Pipeline([("features", FeatureUnion(parts)), ("clf", _classifier(kind))])


def _records_to_norm_text(X) -> List[str]:
    return [normalize(r.get("text", "")).text for r in _as_records(X)]


def build_sarcasm_model(kind: str = "lr", use_text: bool = True, use_incongruity: bool = True,
                        lid: Optional[LanguageIdentifier] = None) -> Pipeline:
    """Input: DataFrame / records with 'text' and optional 'rating'. Output: 0/1."""
    parts = []
    if use_text:
        parts.append(("text", Pipeline([
            ("norm", FunctionTransformer(_records_to_norm_text, feature_names_out="one-to-one")),
            ("tfidf", _text_features(min_df=2)),
        ])))
    if use_incongruity:
        parts.append(("incong", Pipeline([
            ("inc", IncongruityFeatures(lid=lid)),
            ("scale", MinMaxScaler()),   # keeps values >= 0 so NB also works
        ])))
    if not parts:
        raise ValueError("sarcasm model needs text and/or incongruity features")
    return Pipeline([("features", FeatureUnion(parts)), ("clf", _classifier(kind))])


def fit_with_balance(model: Pipeline, X, y, kind: str) -> Pipeline:
    """NB has no class_weight, so we pass balanced sample weights instead."""
    if kind == "nb":
        w = compute_sample_weight("balanced", y)
        return model.fit(X, y, clf__sample_weight=w)
    return model.fit(X, y)


# ----------------------------------------------------------------------------
# the full ABSA system
# ----------------------------------------------------------------------------
@dataclass
class AspectResult:
    aspect: str
    window: str
    sentiment_raw: str
    proba: Dict[str, float]
    sentiment_final: str
    corrected: bool = False


@dataclass
class ReviewResult:
    raw: str
    normalized: str
    tokens: List[str]
    langs: List[str]
    cmi: float
    aspects: List[AspectResult]
    sarcastic: bool
    sarcasm_proba: float
    incongruity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["aspects"] = [a.__dict__ for a in self.aspects]
        return d


class HinglishABSA:
    def __init__(self, lid: Optional[LanguageIdentifier] = None, window_k: int = 5,
                 use_window: bool = True, stop_at_clause: bool = True,
                 sentiment_kind: str = "lr", sarcasm_kind: str = "lr",
                 use_sarcasm: bool = True, sarcasm_threshold: float = 0.5,
                 use_lexicon_feats: bool = True):
        self.lid = lid or LanguageIdentifier()
        self.window_k = window_k
        self.use_window = use_window
        self.stop_at_clause = stop_at_clause
        self.sentiment_kind = sentiment_kind
        self.sarcasm_kind = sarcasm_kind
        self.use_sarcasm = use_sarcasm
        self.sarcasm_threshold = sarcasm_threshold
        self.use_lexicon_feats = use_lexicon_feats
        self.sentiment_model: Optional[Pipeline] = None
        self.sarcasm_model: Optional[Pipeline] = None
        self.meta: dict = {}

    # ---------------- data preparation ----------------
    def build_instances(self, reviews: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        """
        One row per (review, gold aspect): the window text for that aspect.
        'found' tells whether the lexicon actually detected that aspect; when it
        did not, we fall back to the full review text so no gold label is lost.
        """
        rev = reviews.set_index("review_id")
        rows = []
        for r in labels.itertuples(index=False):
            text = rev.loc[r.review_id, "text"]
            toks = normalize(text).tokens
            wins = aspect_windows(toks, k=self.window_k, stop_at_clause=self.stop_at_clause,
                                  use_window=self.use_window)
            found = r.aspect in wins
            window = wins[r.aspect] if found else " ".join(toks)
            rows.append({"review_id": r.review_id, "aspect": r.aspect,
                         "sentiment": r.sentiment, "window": window, "found": found,
                         "text": text})
        return pd.DataFrame(rows)

    # ---------------- training ----------------
    def fit(self, reviews: pd.DataFrame, labels: pd.DataFrame,
            extra_sarcasm: Optional[pd.DataFrame] = None) -> "HinglishABSA":
        """extra_sarcasm: optional DataFrame(text, sarcastic) from a public corpus that is
        added to the sarcasm training set (transfer); never used for aspect sentiment."""
        inst = self.build_instances(reviews, labels)
        self.sentiment_model = build_sentiment_model(self.sentiment_kind, self.use_lexicon_feats)
        fit_with_balance(self.sentiment_model, inst["window"].tolist(), inst["sentiment"].tolist(),
                         self.sentiment_kind)
        if self.use_sarcasm and "sarcastic" in reviews.columns:
            cols = ["text"] + (["rating"] if "rating" in reviews.columns else [])
            X = reviews[cols]
            y = reviews["sarcastic"].astype(int).tolist()
            if extra_sarcasm is not None:
                X = pd.concat([X, extra_sarcasm.reindex(columns=cols)], ignore_index=True)
                y = y + extra_sarcasm["sarcastic"].astype(int).tolist()
            self.sarcasm_model = build_sarcasm_model(self.sarcasm_kind, lid=self.lid)
            fit_with_balance(self.sarcasm_model, X, y, self.sarcasm_kind)
            self.meta["sarcasm_extra_train"] = 0 if extra_sarcasm is None else int(len(extra_sarcasm))
        self.meta.update({"n_reviews": int(len(reviews)), "n_instances": int(len(inst)),
                          "aspect_found_rate": float(inst["found"].mean())})
        return self

    # ---------------- inference ----------------
    def _sentiment(self, window: str) -> tuple[str, Dict[str, float]]:
        proba = self.sentiment_model.predict_proba([window])[0]
        classes = list(self.sentiment_model.classes_)
        probs = {c: float(p) for c, p in zip(classes, proba)}
        return classes[int(np.argmax(proba))], probs

    def sarcasm_proba(self, text: str, rating: Optional[float] = None) -> float:
        if not (self.use_sarcasm and self.sarcasm_model is not None):
            return 0.0
        X = pd.DataFrame([{"text": text, "rating": rating if rating is not None else np.nan}])
        proba = self.sarcasm_model.predict_proba(X)[0]
        classes = list(self.sarcasm_model.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0

    @staticmethod
    def correct_polarity(sentiment: str, sarcastic: bool) -> tuple[str, bool]:
        """
        Polarity correction rule: sarcasm in reviews is almost always mock praise,
        so a POSITIVE surface sentiment under sarcasm is flipped to NEGATIVE.
        Neutral and negative are left unchanged (a sarcastic complaint is still
        a complaint).
        """
        if sarcastic and sentiment == "positive":
            return "negative", True
        return sentiment, False

    def predict_review(self, text: str, rating: Optional[float] = None) -> ReviewResult:
        norm = normalize(text)
        toks = norm.tokens
        lid_res = self.lid.analyse(toks)
        wins = aspect_windows(toks, k=self.window_k, stop_at_clause=self.stop_at_clause,
                              use_window=self.use_window)
        p_sarc = self.sarcasm_proba(text, rating)
        sarcastic = p_sarc >= self.sarcasm_threshold
        from .features import incongruity_report
        inc = incongruity_report(text, rating, self.lid) if self.use_sarcasm else {}

        aspects: List[AspectResult] = []
        for aspect, window in wins.items():
            lab, probs = self._sentiment(window)
            final, corrected = self.correct_polarity(lab, sarcastic)
            aspects.append(AspectResult(aspect, window, lab, probs, final, corrected))
        return ReviewResult(raw=text, normalized=norm.text, tokens=toks, langs=lid_res.langs,
                            cmi=lid_res.cmi, aspects=aspects, sarcastic=bool(sarcastic),
                            sarcasm_proba=p_sarc, incongruity=inc)

    def predict_instances(self, inst: pd.DataFrame) -> List[str]:
        return list(self.sentiment_model.predict(inst["window"].tolist()))

    # ---------------- persistence ----------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "HinglishABSA":
        return joblib.load(path)
