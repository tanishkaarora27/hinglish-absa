"""
Convert the filled labelling sheet into the two files the notebook reads:
  data/labelled/reviews.csv        review_id, text, rating, sarcastic, source
  data/labelled/aspect_labels.csv  review_id, aspect, sentiment

Run:  python scripts/prepare_labelled_data.py [--sheet data/labelling_sheet.csv]
Also prints inter-annotator agreement (Cohen's kappa) if a second sheet is
given with --sheet2 (same rows labelled by a second person).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from absa.aspects import ASPECTS  # noqa: E402

MAP = {"pos": "positive", "positive": "positive", "p": "positive", "+": "positive",
       "neg": "negative", "negative": "negative", "n": "negative", "-": "negative",
       "neu": "neutral", "neutral": "neutral", "0": "neutral", "=": "neutral"}


def to_long(sheet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheet = sheet.copy()
    sheet["review_id"] = sheet["review_id"].astype(str)
    rows = []
    for r in sheet.itertuples(index=False):
        for a in ASPECTS:
            v = str(getattr(r, a, "")).strip().lower()
            if v in MAP:
                rows.append({"review_id": r.review_id, "aspect": a, "sentiment": MAP[v]})
    labels = pd.DataFrame(rows)
    labelled_ids = set(labels.review_id)
    reviews = sheet[sheet.review_id.isin(labelled_ids)][["review_id", "text", "rating", "sarcastic"]].copy()
    reviews["sarcastic"] = pd.to_numeric(reviews["sarcastic"], errors="coerce").fillna(0).astype(int)
    reviews["source"] = sheet.get("app", "play_store")
    return reviews, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(ROOT / "data" / "labelling_sheet.csv"))
    ap.add_argument("--sheet2", default=None, help="optional second annotator's sheet for agreement")
    args = ap.parse_args()
    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    reviews, labels = to_long(sheet)
    out = ROOT / "data" / "labelled"; out.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(out / "reviews.csv", index=False); labels.to_csv(out / "aspect_labels.csv", index=False)
    print(f"{len(reviews)} labelled reviews, {len(labels)} aspect instances, "
          f"{reviews.sarcastic.sum()} sarcastic -> {out}")
    print(pd.crosstab(labels.aspect, labels.sentiment, margins=True))
    if len(reviews) < 1500:
        print(f"[note] {len(reviews)} < 1500 reviews: the assignment asks for 1,500-3,000; label more if you can.")

    if args.sheet2:
        from sklearn.metrics import cohen_kappa_score
        s2 = pd.read_csv(args.sheet2, dtype=str, keep_default_na=False)
        _, l2 = to_long(s2)
        m = labels.merge(l2, on=["review_id", "aspect"], suffixes=("_a", "_b"))
        print(f"\nagreement on {len(m)} shared (review, aspect) labels:")
        print("  sentiment Cohen's kappa:", round(cohen_kappa_score(m.sentiment_a, m.sentiment_b), 3))
        r2 = s2[s2.review_id.isin(reviews.review_id)].set_index("review_id")
        both = reviews.set_index("review_id").join(r2[["sarcastic"]], rsuffix="_b", how="inner")
        both["sarcastic_b"] = pd.to_numeric(both.sarcastic_b, errors="coerce").fillna(0).astype(int)
        print("  sarcasm   Cohen's kappa:", round(cohen_kappa_score(both.sarcastic, both.sarcastic_b), 3))


if __name__ == "__main__":
    main()
