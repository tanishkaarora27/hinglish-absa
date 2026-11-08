"""
Download the PUBLIC Hinglish resources used by this project and convert them
into tidy CSV/TXT files under data/external/.

Run:  python scripts/download_external_data.py
Needs: git on PATH and internet. Takes ~1-2 minutes.

Resources (all verified to exist on 2026-11-07):
 1. Swami et al. (2018) "A Corpus of English-Hindi Code-Mixed Tweets for Sarcasm
    Detection" - 5,250 tweets, YES/NO sarcasm labels, word-level language tags.
    Repo: https://github.com/sahilswami96/SarcasmDetection_CodeMixed  (GPL-3.0)
 2. L3Cube-HingLID (Nayak & Joshi, WILDRE 2022) - word-level Hindi/English LID
    corpus (train/validation/test). Repo: https://github.com/l3cube-pune/code-mixed-nlp
    License: CC BY-NC-SA 4.0
 3. Shah & Maurya (ICON 2021) "How effective is incongruity?" - 166K hashtag-
    labelled (distantly supervised) Hinglish sarcasm tweets, train/valid/test.
    Repo: https://github.com/likemycode/codemix   (no license file in repo)
 4. Mamta & Ekbal (Computer Speech & Language 2025) ABSA-Mix - Hinglish ABSA
    corpus (Devanagari Hindi + English, restaurant/laptop). Repo:
    https://github.com/20118/ABSA-MIX   (no license file in repo)

Only (1) and (2) are needed to run the notebook. (3) and (4) are optional
extra experiments / related-work material.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
REPOS = {
    "swami_sarcasm": "https://github.com/sahilswami96/SarcasmDetection_CodeMixed",
    "l3cube": "https://github.com/l3cube-pune/code-mixed-nlp",
    "shah_maurya_codemix": "https://github.com/likemycode/codemix",
    "absa_mix": "https://github.com/20118/ABSA-MIX",
}


def clone(name: str, url: str) -> Path:
    dest = EXT / name
    if dest.exists():
        print(f"[skip] {name} already present")
        return dest
    print(f"[clone] {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(dest)], check=True)
    return dest


def parse_swami(repo: Path) -> None:
    """Build swami_sarcasm.csv with columns tweet_id,text,sarcastic (1/0)."""
    d = repo / "Dataset"
    texts = {}
    lines = (d / "Sarcasm_tweets.txt").read_text(encoding="utf-8", errors="ignore").split("\n")
    i = 0
    while i < len(lines):
        if re.fullmatch(r"\d+", lines[i].strip()):
            tid = lines[i].strip()
            j = i + 1
            buf = []
            while j < len(lines) and not re.fullmatch(r"\d+", lines[j].strip()) and lines[j].strip() != "":
                buf.append(lines[j].strip())
                j += 1
            texts[tid] = " ".join(buf)
            i = j
        else:
            i += 1
    truth = {}
    tl = (d / "Sarcasm_tweet_truth.txt").read_text(encoding="utf-8", errors="ignore").split("\n")
    for a, b in zip(tl, tl[1:]):
        if re.fullmatch(r"\d+", a.strip()) and b.strip() in ("YES", "NO"):
            truth[a.strip()] = 1 if b.strip() == "YES" else 0
    out = EXT / "swami_sarcasm.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "text", "sarcastic"])
        n = 0
        for tid, lab in truth.items():
            if tid in texts and texts[tid]:
                w.writerow([tid, texts[tid], lab])
                n += 1
    print(f"[ok] swami_sarcasm.csv: {n} tweets, {sum(truth.values())} sarcastic")
    # keep the LID file next to it
    (EXT / "swami_lid_tags.txt").write_bytes((d / "Sarcasm_tweets_with_language.txt").read_bytes())


def main() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    only_core = "--core-only" in sys.argv
    for name, url in REPOS.items():
        if only_core and name not in ("swami_sarcasm", "l3cube"):
            continue
        try:
            repo = clone(name, url)
        except subprocess.CalledProcessError as e:
            print(f"[warn] could not clone {name}: {e}")
            continue
        if name == "swami_sarcasm":
            parse_swami(repo)
        elif name == "l3cube":
            src = repo / "L3Cube-HingLID"
            for part in ("train", "validation", "test"):
                (EXT / f"l3cube_hinglid_{part}.txt").write_bytes((src / f"{part}.txt").read_bytes())
            print("[ok] L3Cube-HingLID copied (train/validation/test)")
        elif name == "shah_maurya_codemix":
            for part in ("train", "valid", "test"):
                (EXT / f"shah_maurya_{part}.csv").write_bytes((repo / f"{part}.csv").read_bytes())
            print("[ok] Shah & Maurya sarcasm CSVs copied")
        elif name == "absa_mix":
            for fn in ("Complete_restaurant_train.csv", "Complete_restaurant_test.csv",
                       "complete_laptop_train.csv", "complete_laptop_test.csv"):
                (EXT / f"absamix_{fn.lower()}").write_bytes((repo / fn).read_bytes())
            print("[ok] ABSA-Mix CSVs copied")
    print("Done. Files in", EXT)


if __name__ == "__main__":
    main()
