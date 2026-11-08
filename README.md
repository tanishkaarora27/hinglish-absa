# Aspect-Based Sentiment Analysis of Hinglish Quick-Commerce Reviews with Sarcasm Detection

B.Tech CSE (AI) lab mini-project. Python + scikit-learn, **CPU only**, runs on any laptop or free Google Colab.

```
normalise → word-level LID + CMI → aspect extraction (lexicon+rules) → aspect windows
→ aspect sentiment (NB / LR / Linear-SVM, char n-gram TF-IDF) → sarcasm (incongruity features)
→ polarity correction → explanation (feature weights + LIME)
```

## Folder map
| Path | What |
|---|---|
| `src/absa/` | the pipeline, one module per stage (`normalize`, `lid`, `lexicon`, `aspects`, `features`, `pipeline`, `explain`, `evaluation`) |
| `notebooks/hinglish_absa.ipynb` | all experiments: baselines, CV, ablations, error analysis, explanations; saves models + `results/` |
| `app.py` | Streamlit live demo |
| `scripts/download_external_data.py` | fetches public Hinglish datasets (L3Cube-HingLID, Swami et al. sarcasm, …) |
| `scripts/scrape_reviews.py` | collects public Google Play reviews of Blinkit / Zepto / Swiggy |
| `scripts/make_labelling_sheet.py` | builds the CSV you label (pre-filled aspect guesses) |
| `scripts/prepare_labelled_data.py` | converts the labelled sheet into `data/labelled/` (+ Cohen's kappa) |
| `data/demo/` | 149 **hand-written seed reviews** – smoke test in DEMO mode; in REAL mode their 20 sarcastic examples are added to sarcasm *training* only (never evaluation) |
| `docs/` | research summary, labelling guideline, PPT content, demo script, viva pack, rubric |
| `results/`, `models/` | produced by the notebook |

## Setup (any OS)
```bash
git clone <your repo>   # or unzip the folder
cd hinglish_absa
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_external_data.py --core-only   # ~1 min, needs git + internet
```
**Google Colab:** upload/unzip the folder, then in a cell: `!pip install -r requirements.txt && !python scripts/download_external_data.py --core-only`, then open the notebook (File → Open) or run `!jupyter nbconvert --to notebook --execute notebooks/hinglish_absa.ipynb`.

## Run in 3 steps
1. **Collect + label your data** (the real project data; ~1,500–3,000 reviews):
   ```bash
   python scripts/scrape_reviews.py --per-app 3000     # -> data/raw/play_reviews_raw.csv
   python scripts/make_labelling_sheet.py --n 2500      # -> data/labelling_sheet.csv (open in Excel/Sheets)
   # label per docs/01_labelling_guidelines.md, then:
   python scripts/prepare_labelled_data.py              # -> data/labelled/reviews.csv + aspect_labels.csv
   ```
   Skip this step to run in **DEMO mode** on the seed data (code check only).
2. **Run all experiments** – open `notebooks/hinglish_absa.ipynb` and *Run All* (≈3 min on CPU in demo mode; ≈15–20 min with 2,500 real reviews — the sarcasm transfer experiments are the slow part). It prints every table, saves figures/CSVs in `results/` and the trained model in `models/`.
3. **Live demo:** `streamlit run app.py` → browser opens at http://localhost:8501. Type a Hinglish review (optionally choose the star rating) → aspects, per-aspect sentiment, sarcasm flag, and the evidence.

## Live demo using clickable link
https://hinglish-absa-eretvxjs3ip9vktdc2dqfa.streamlit.app/

## Honest-reporting rule built into the code
The notebook detects whether `data/labelled/` exists. If not, every output is tagged `[DEMO-SYNTHETIC]` and the app shows a warning banner. Only two experiments always use real data: the LID model (L3Cube-HingLID + Swami cross-check) and the sarcasm benchmark on the Swami et al. (2018) corpus.

## Credits / data licences
* L3Cube-HingLID – Nayak & Joshi (2022), CC BY-NC-SA 4.0 – https://github.com/l3cube-pune/code-mixed-nlp
* Swami et al. (2018) sarcasm corpus – repo licensed GPL-3.0 – https://github.com/sahilswami96/SarcasmDetection_CodeMixed
* Shah & Maurya (2021) 166K sarcasm tweets (optional) – https://github.com/likemycode/codemix (no licence file; research use, cite the paper)
* Mamta & Ekbal (2025) ABSA-Mix (optional, Devanagari) – https://github.com/20118/ABSA-MIX (no licence file; cite the paper)
* google-play-scraper (MIT) – https://pypi.org/project/google-play-scraper/
* LIME – Ribeiro, Singh & Guestrin (2016), BSD-2.
