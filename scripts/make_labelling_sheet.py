"""
Turn raw scraped reviews into a labelling sheet (CSV you open in Excel /
Google Sheets). Keeps mostly-Hinglish reviews, drops near-duplicates and very
short texts, balances ratings a little, and PRE-FILLS the aspect columns with
the rule-based extractor's guess so annotators only have to correct.

Run:  python scripts/make_labelling_sheet.py --n 2500
In :  data/raw/play_reviews_raw.csv
Out:  data/labelling_sheet.csv   (one row per review, one column per aspect)

Columns to fill (see docs/01_labelling_guidelines.md):
  delivery_speed, product_quality, packaging, price, delivery_partner, app_support, general
      -> one of: pos | neg | neu | (blank = aspect not mentioned)
  sarcastic -> 1 or 0
  notes     -> free text (optional)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from absa.aspects import ASPECTS, detect_aspects          # noqa: E402
from absa.lid import LanguageIdentifier, read_l3cube_lid  # noqa: E402
from absa.normalize import normalize                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--min-hindi-frac", type=float, default=0.15,
                    help="keep reviews with at least this fraction of Hindi tokens (code-mixed filter)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = pd.read_csv(ROOT / "data" / "raw" / "play_reviews_raw.csv")
    raw["text"] = raw["text"].astype(str)
    raw = raw[raw.text.str.split().str.len() >= args.min_words]
    raw["key"] = raw.text.str.lower().str.replace(r"[^a-z]", "", regex=True)
    raw = raw.drop_duplicates("key")

    # language filter: train a quick LID (or load the notebook's one)
    lid_path = ROOT / "models" / "lid.joblib"
    if lid_path.exists():
        lid = LanguageIdentifier.load(str(lid_path))
    else:
        w, l = read_l3cube_lid(str(ROOT / "data" / "external" / "l3cube_hinglid_train.txt"), max_tokens=200_000)
        lid = LanguageIdentifier().fit(w, l)
    def hindi_frac(t):
        toks = normalize(t).tokens
        return lid.analyse(toks).frac_hi if toks else 0.0
    raw["hindi_frac"] = raw.text.map(hindi_frac)
    mixed = raw[raw.hindi_frac >= args.min_hindi_frac]
    print(f"raw {len(raw)} -> code-mixed {len(mixed)} (>= {args.min_hindi_frac:.0%} Hindi tokens)")

    # mild balancing over star ratings so 1-star complaints do not dominate
    per_rating = max(args.n // 5, 1)
    parts = [g.sample(min(len(g), per_rating), random_state=args.seed) for _, g in mixed.groupby("rating")]
    sample = pd.concat(parts)
    if len(sample) < args.n:  # top up from the remainder
        rest = mixed.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(len(rest), args.n - len(sample)), random_state=args.seed)])
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    sheet = sample[["app", "review_id", "rating", "date", "text"]].copy()
    for a in ASPECTS:
        sheet[a] = ""
    # pre-fill: mark detected aspects with "?" so the annotator sees the guess and replaces it
    for i, t in enumerate(sheet.text):
        for a in detect_aspects(normalize(t).tokens):
            sheet.loc[i, a] = "?"
    sheet["sarcastic"] = ""
    sheet["notes"] = ""
    out = ROOT / "data" / "labelling_sheet.csv"
    sheet.to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows emoji correctly
    print(f"wrote {len(sheet)} rows -> {out}\nFill pos/neg/neu per aspect column, sarcastic 1/0, then run prepare_labelled_data.py")


if __name__ == "__main__":
    main()
