"""
Evaluation helpers: stratified (group-aware) cross-validation, metrics,
per-aspect results, confusion matrices, ablations and error tables.

Why stratified GROUP K-fold for aspect sentiment?
    One review can produce several (aspect, sentiment) instances. If two
    instances of the same review land in different folds, the model sees part
    of the test review during training (leakage). GroupKFold keeps all
    instances of a review together; the stratified variant keeps class
    proportions similar across folds, which matters with imbalance.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .aspects import ASPECTS, detect_aspects
from .normalize import normalize
from .pipeline import fit_with_balance

SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def summarise(y_true, y_pred, labels=None) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                 labels=labels, zero_division=0)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {"accuracy": round(acc, 4), "precision_macro": round(p, 4),
            "recall_macro": round(r, 4), "f1_macro": round(f, 4)}


# --------------------------------------------------------------------------
# aspect sentiment
# --------------------------------------------------------------------------
def cv_sentiment(instances: pd.DataFrame, build_fn: Callable[[], object], kind: str,
                 n_splits: int = 5, seed: int = 42) -> Dict:
    """Out-of-fold predictions for aspect-sentiment instances (group = review)."""
    X = instances["window"].tolist()
    y = instances["sentiment"].to_numpy()
    groups = instances["review_id"].to_numpy()
    n_splits = min(n_splits, int(pd.Series(y).value_counts().min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    fold_f1 = []
    for tr, te in cv.split(X, y, groups):
        model = build_fn()
        fit_with_balance(model, [X[i] for i in tr], y[tr].tolist(), kind)
        pred = model.predict([X[i] for i in te])
        oof[te] = pred
        fold_f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
    res = summarise(y, oof, labels=SENTIMENT_ORDER)
    res.update({"f1_macro_std": round(float(np.std(fold_f1)), 4), "n_splits": n_splits,
                "oof_pred": oof})
    return res


def per_aspect_table(instances: pd.DataFrame, oof_pred) -> pd.DataFrame:
    df = instances.assign(pred=oof_pred)
    rows = []
    for aspect, g in df.groupby("aspect"):
        s = summarise(g["sentiment"], g["pred"], labels=SENTIMENT_ORDER)
        rows.append({"aspect": aspect, "n": len(g), **s})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def confusion(y_true, y_pred, labels=SENTIMENT_ORDER) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def report_text(y_true, y_pred, labels=SENTIMENT_ORDER) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3)


# --------------------------------------------------------------------------
# sarcasm (review level)
# --------------------------------------------------------------------------
def cv_sarcasm(reviews: pd.DataFrame, build_fn: Callable[[], object], kind: str,
               n_splits: int = 5, seed: int = 42,
               extra_train: Optional[pd.DataFrame] = None) -> Dict:
    """
    Review-level sarcasm CV. `extra_train` (e.g. the Swami et al. tweet corpus) is
    appended to EVERY training fold but never tested on -> "transfer" setting:
    the public corpus supplies sarcastic examples that the small review set lacks.
    """
    cols = ["text"] + (["rating"] if "rating" in reviews.columns else [])
    X = reviews[cols].reset_index(drop=True)
    y = reviews["sarcastic"].astype(int).to_numpy()
    n_splits = min(n_splits, int(pd.Series(y).value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y))
    if extra_train is not None:
        extra_X = extra_train.reindex(columns=cols)      # rating -> NaN for tweets
        extra_y = extra_train["sarcastic"].astype(int).to_numpy()
    for tr, te in cv.split(X, y):
        model = build_fn()
        Xtr, ytr = X.iloc[tr], y[tr]
        if extra_train is not None:
            Xtr = pd.concat([Xtr, extra_X], ignore_index=True)
            ytr = np.concatenate([ytr, extra_y])
        fit_with_balance(model, Xtr, ytr.tolist(), kind)
        oof[te] = model.predict(X.iloc[te])
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            oof_proba[te] = model.predict_proba(X.iloc[te])[:, classes.index(1)] if 1 in classes else 0
    p, r, f, _ = precision_recall_fscore_support(y, oof, average="binary", zero_division=0)
    res = {"accuracy": round(float(np.mean(y == oof)), 4), "precision_sarcastic": round(p, 4),
           "recall_sarcastic": round(r, 4), "f1_sarcastic": round(f, 4),
           "f1_macro": round(f1_score(y, oof, average="macro", zero_division=0), 4),
           "n_splits": n_splits, "oof_pred": oof, "oof_proba": oof_proba}
    return res


# --------------------------------------------------------------------------
# aspect extraction (lexicon + rules) against gold aspects
# --------------------------------------------------------------------------
def aspect_extraction_report(reviews: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    gold = labels.groupby("review_id")["aspect"].apply(set).to_dict()
    rows = []
    for r in reviews.itertuples(index=False):
        pred = set(detect_aspects(normalize(r.text).tokens))
        rows.append({"review_id": r.review_id, "gold": gold.get(r.review_id, set()), "pred": pred})
    out = []
    for a in ASPECTS:
        tp = sum(a in x["gold"] and a in x["pred"] for x in rows)
        fp = sum(a not in x["gold"] and a in x["pred"] for x in rows)
        fn = sum(a in x["gold"] and a not in x["pred"] for x in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        out.append({"aspect": a, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f, 3)})
    df = pd.DataFrame(out)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    df.loc[len(df)] = {"aspect": "MICRO-AVG", "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                       "precision": round(p, 3), "recall": round(rc, 3),
                       "f1": round(2 * p * rc / (p + rc) if p + rc else 0, 3)}
    return df


# --------------------------------------------------------------------------
# end-to-end with / without sarcasm correction
# --------------------------------------------------------------------------
def apply_correction(sent_pred: List[str], sarcastic_by_review: Dict, review_ids: List) -> List[str]:
    out = []
    for s, rid in zip(sent_pred, review_ids):
        if sarcastic_by_review.get(rid, 0) and s == "positive":
            out.append("negative")
        else:
            out.append(s)
    return out


def error_table(instances: pd.DataFrame, oof_pred, n: int = 25) -> pd.DataFrame:
    df = instances.assign(pred=oof_pred)
    err = df[df["sentiment"] != df["pred"]].copy()
    return err[["review_id", "aspect", "sentiment", "pred", "window", "text", "found"]].head(n)
