# Rubric-style self-check against the assignment brief

| # | Requirement in the brief | Where it is met | Status |
|---|---|---|---|
| 1 | 4–6 real papers (prefer 2022–26) with authors, year, venue, work, limitation | `docs/00_research_summary.md` §1 — 6 papers (2018, 2020, 2021, 2022, 2024/25, 2025) + 4 supporting | ✅ verified live on 7 Nov 2026 |
| 2 | Real public datasets verified (exist, licence, size) | `00_research_summary.md` §2 — L3Cube-HingLID (CC BY-NC-SA, parsed), Swami 2018 (GPL-3 repo, 5,250/504), SentiMix (Zenodo metadata; not downloadable from sandbox), Shah & Maurya 166K, ABSA-Mix | ✅ (SentiMix metadata only — stated) |
| 3 | Legal/ethical quick-commerce review collection method + labelling guideline | `scripts/scrape_reviews.py` (google-play-scraper, ethics note), `scripts/make_labelling_sheet.py`, `docs/01_labelling_guidelines.md`; **2,500 real reviews scraped and labelled → `data/labelled/`** | ✅ |
| 4 | CPU-only, free Colab, Python + sklearn + pandas, lightweight libs | `requirements.txt`; full notebook runs in ~18 min CPU on the real data (3 min in demo mode) | ✅ |
| 5 | Transformer only as optional, justified comparison | notebook §12, `02_design_and_concepts.md` §4 | ✅ |
| 6 | Pipeline: normalisation → LID + CMI → aspect extraction (lexicon+rules) → aspect-conditioned windows → sentiment → incongruity sarcasm → polarity correction | `src/absa/normalize.py, lid.py, aspects.py, features.py, pipeline.py` | ✅ |
| 7 | Advanced concept 1: aspect-conditioned context windows, explained | `aspects.py` docstring, `02_design` §3.1, viva Q41–42 | ✅ |
| 8 | Advanced concept 2: sentiment-incongruity sarcasm, explained | `features.py` docstring, `02_design` §3.2, viva Q30–34 | ✅ |
| 9 | Advanced concept 3: explainability (LIME / feature weights) | `explain.py`, app "Why?" tabs, notebook §9 | ✅ both methods |
| 10 | ≥2–3 baselines with char n-gram TF-IDF, imbalance handling, stratified CV | notebook §5: ComplementNB / LR / Linear-SVM; balanced weights; StratifiedGroupKFold | ✅ |
| 11 | Precision, recall, macro-F1, per-aspect, confusion matrix | notebook §5 + `results/sentiment_baselines.csv, per_aspect.csv, fig_confusion.png` | ✅ |
| 12 | Ablation with/without sarcasm handling and with/without windows | notebook §6 & §8; `results/ablation_windows*.csv, ablation_sarcasm.csv` | ✅ |
| 13 | Error analysis with real misclassified examples | notebook §10; `results/errors_sentiment.csv` (431 errors, 40 categorised), `errors_sarcasm.csv` | ✅ real examples |
| 14 | Results only from code that runs; no fabricated metrics | Every number in the PPT / viva docs comes from the executed notebook in REAL mode (`results/metrics.json`); the sarcasm-data shortage is reported, not hidden | ✅ |
| 15 | Runnable notebook + live demo (Streamlit) + requirements + setup instructions | `notebooks/hinglish_absa.ipynb` (executed), `app.py` (served, HTTP 200 in test), `requirements.txt`, `README.md` | ✅ |
| 16 | PPT content: all 10 required sections, visuals, speaker notes, not text-heavy | `docs/03_ppt_content.md` — 17 slides incl. Collaboration (solo + team-of-4 variants) | ✅ |
| 17 | 3–4 minute demo script | `docs/04_demo_script.md` (timed, with fallbacks) | ✅ |
| 18 | ≥40 viva Q&A grouped by topic | `docs/05_viva_pack.md` §A — 46 questions, 9 groups | ✅ |
| 19 | ≥15 tough cross-questions with follow-ups | `05_viva_pack.md` §B — 16 | ✅ |
| 20 | Honest limitations (data size, label noise, sarcasm ambiguity, domain, lexicon, spelling) + how to acknowledge | `05_viva_pack.md` §C | ✅ |
| 21 | One-liners "why not deep learning" | `05_viva_pack.md` §D, `02_design` §5 | ✅ |
| 22 | Organised so it can become the written report later | `docs/00` (lit review) → `02` (method) → notebook/`results` (experiments) → `05 §C` (limitations) → `03` slide 15 (conclusion/future) | ✅ |

## Still on you
1. Take two screenshots of the app (multi-aspect example; sarcastic example with rating 1, "Why?" tab open) for slides 2, 11 and 14.
2. Decide solo vs team and delete the unused Collaboration variant on slide 17.
3. Build the actual slide deck from `03_ppt_content.md` (or ask for it).
4. Optional: a second annotator on 200 rows → `prepare_labelled_data.py --sheet2` gives Cohen's kappa for slide 8.
