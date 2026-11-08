"""
Explainability (advanced concept #3): LIME + feature-weight inspection.

Two complementary ways to answer "WHY did the model say this?":

1. Feature-weight inspection (exact, instant, linear models only)
   For Logistic Regression / Linear SVM the score for class c is
       score_c(x) = b_c + sum_i w_{c,i} * x_i
   so each feature's contribution to the decision is simply w_{c,i} * x_i.
   We compute this for the actual input and show the top positive / negative
   contributions (e.g. the 4-gram "bakw" pushed towards NEGATIVE by +0.31).
   It is exact for linear models - no approximation.

2. LIME (model-agnostic, approximate)
   LIME (Ribeiro et al., 2016) perturbs the input text by randomly removing
   words, asks the model for probabilities on each perturbed copy, and fits a
   small linear model locally to see which words mattered. It works for ANY
   classifier (including Naive Bayes or a transformer) and explains at WORD
   level, which is easier to read than char n-grams. It is slower (hundreds of
   model calls) so we use num_samples ~300 in the demo.

Both are used in the notebook; the Streamlit app shows feature contributions
by default and LIME on demand.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------
# helpers to reach inside the pipelines
# --------------------------------------------------------------------------
def feature_names(model: Pipeline) -> np.ndarray:
    return model.named_steps["features"].get_feature_names_out()


def class_weight_matrix(model: Pipeline) -> Tuple[np.ndarray, List]:
    """Return (W, classes) with W shape (n_classes, n_features) for lr / svm / nb."""
    clf = model.named_steps["clf"]
    classes = list(clf.classes_)
    if hasattr(clf, "coef_"):                      # LogisticRegression / LinearSVC
        W = np.asarray(clf.coef_)
    elif hasattr(clf, "calibrated_classifiers_"):   # CalibratedClassifierCV(LinearSVC)
        W = np.mean([np.asarray(c.estimator.coef_) for c in clf.calibrated_classifiers_], axis=0)
    elif hasattr(clf, "feature_log_prob_"):         # ComplementNB
        W = np.asarray(clf.feature_log_prob_)
        W = W - W.mean(axis=0, keepdims=True)       # relative evidence per class
    else:
        raise TypeError("classifier has no inspectable weights")
    if W.shape[0] == 1 and len(classes) == 2:       # binary case -> expand to 2 rows
        W = np.vstack([-W, W])
    return W, classes


def top_features(model: Pipeline, n: int = 15) -> pd.DataFrame:
    """Global view: strongest features per class (what the model learned)."""
    W, classes = class_weight_matrix(model)
    names = feature_names(model)
    rows = []
    for ci, c in enumerate(classes):
        idx = np.argsort(W[ci])[::-1][:n]
        for rank, i in enumerate(idx, 1):
            rows.append({"class": c, "rank": rank, "feature": names[i], "weight": float(W[ci, i])})
    return pd.DataFrame(rows)


def contributions(model: Pipeline, X_one, target_class=None, n: int = 10) -> pd.DataFrame:
    """
    Local view for ONE input: per-feature contribution w * x towards target_class
    (default = predicted class). X_one is a single raw input as the pipeline
    expects it (window string for sentiment, DataFrame row for sarcasm).
    """
    feats = model.named_steps["features"]
    x = feats.transform([X_one] if not isinstance(X_one, pd.DataFrame) else X_one)
    x = x.toarray()[0] if sp.issparse(x) else np.asarray(x)[0]
    W, classes = class_weight_matrix(model)
    if target_class is None:
        target_class = model.predict([X_one] if not isinstance(X_one, pd.DataFrame) else X_one)[0]
    ci = classes.index(target_class)
    contrib = W[ci] * x
    names = feature_names(model)
    nz = np.nonzero(contrib)[0]
    order = nz[np.argsort(np.abs(contrib[nz]))[::-1]][:n]
    return pd.DataFrame({"feature": names[order], "value": x[order],
                         "weight": W[ci, order], "contribution": contrib[order],
                         "towards": target_class})


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------
def lime_explain(model: Pipeline, text: str, num_features: int = 8,
                 num_samples: int = 300, class_names: Optional[Sequence[str]] = None):
    """Word-level LIME explanation for a text classifier pipeline with predict_proba."""
    from lime.lime_text import LimeTextExplainer
    classes = list(model.classes_) if class_names is None else list(class_names)
    explainer = LimeTextExplainer(class_names=[str(c) for c in classes],
                                  split_expression=r"\s+", bow=True, random_state=0)
    pred = model.predict([text])[0]
    label = classes.index(pred)
    exp = explainer.explain_instance(text, model.predict_proba, labels=[label],
                                     num_features=num_features, num_samples=num_samples)
    return pred, exp.as_list(label=label)
