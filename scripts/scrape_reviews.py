"""
Collect PUBLIC Google Play reviews of quick-commerce apps with the
`google-play-scraper` Python library (pip install google-play-scraper).

Run:  python scripts/scrape_reviews.py --per-app 10000
      python scripts/scrape_reviews.py --per-app 10000 --sort relevant   # second pass, merged
Out:  data/raw/play_reviews_raw.csv   (new runs are MERGED into it, duplicates dropped)

Why this is a legitimate collection method (viva answer):
  * Reviews on Google Play are public, posted by users for everyone to read.
  * We call the same public endpoint the Play Store web page uses; no login,
    no scraping of private data, no bypassing of any access control.
  * We store only review text, star rating, date and app name. We do NOT
    store user names or any profile data (the library returns them; we drop
    them before saving) - this keeps the dataset free of personal data.
  * We throttle requests (a pause between pages) and collect a small sample
    for academic, non-commercial research. Please also read Google Play's
    Terms of Service yourself and cite the library in the report.

Package IDs (verify on the Play Store URL ?id=... before running):
  Blinkit            com.grofers.customerapp
  Zepto              com.zeptoconsumerapp
  Swiggy (Instamart lives inside the Swiggy app)  in.swiggy.android
Edit APPS below if an ID has changed.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from google_play_scraper import Sort, reviews

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "play_reviews_raw.csv"
FIELDS = ["app", "review_id", "text", "rating", "date", "thumbs_up"]

APPS = {
    "blinkit": "com.grofers.customerapp",
    "zepto": "com.zeptoconsumerapp",
    "swiggy_instamart": "in.swiggy.android",
}
SORTS = {"newest": Sort.NEWEST, "relevant": Sort.MOST_RELEVANT, "rating": Sort.RATING}


def collect(app_name: str, app_id: str, want: int, lang: str, country: str, pause: float, sort):
    got, token, batch = [], None, 200
    while len(got) < want:
        result, token = reviews(app_id, lang=lang, country=country, sort=sort,
                                count=min(batch, want - len(got)), continuation_token=token)
        for r in result:
            got.append({"app": app_name, "review_id": r.get("reviewId"),
                        "text": (r.get("content") or "").replace("\n", " ").strip(),
                        "rating": r.get("score"), "date": str(r.get("at"))[:10],
                        "thumbs_up": r.get("thumbsUpCount")})
        print(f"  {app_name}: {len(got)} reviews")
        if token is None or not result:
            print(f"  {app_name}: store returned no more pages")
            break
        time.sleep(pause)
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-app", type=int, default=10000, help="reviews to fetch per app")
    ap.add_argument("--lang", default="en", help="Play Store language code; 'en' returns mostly Hinglish/English for India")
    ap.add_argument("--country", default="in")
    ap.add_argument("--sort", default="newest", choices=list(SORTS), help="run twice with different sorts to get more unique reviews")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between pages (be polite)")
    ap.add_argument("--fresh", action="store_true", help="discard the existing raw file instead of merging into it")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if OUT.exists() and not args.fresh:
        with open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["review_id"]] = r
        print(f"merging into existing file with {len(existing)} reviews")

    new = 0
    for name, app_id in APPS.items():
        print(f"collecting {name} ({app_id}) sort={args.sort} ...")
        try:
            for r in collect(name, app_id, args.per_app, args.lang, args.country, args.pause, SORTS[args.sort]):
                if r["text"] and r["review_id"] not in existing:
                    existing[r["review_id"]] = r
                    new += 1
        except Exception as e:  # keep going if one app fails
            print(f"  [warn] {name} failed: {e}")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(existing.values())
    print(f"added {new} new reviews; file now has {len(existing)} unique reviews -> {OUT}")
    print("Next: python scripts/make_labelling_sheet.py --n 2500")


if __name__ == "__main__":
    main()
