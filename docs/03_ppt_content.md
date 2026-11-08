# PPT — slide-by-slide content

Style: one idea per slide, ≤ 5 bullets, big visual. 16 slides ≈ 10–12 min. All numbers below come from the executed notebook on the real labelled data (`results/`). Speaker notes are what you say, not what you show.

---
### Slide 1 — Problem Title
**Title:** Aspect-Based Sentiment Analysis of Hinglish Quick-Commerce Reviews with Sarcasm Detection
**Sub-title:** CSE (AI) Lab Mini-Project · [names, roll nos, section] · [guide name] · [date]
**Visual:** three app-review bubbles in Hinglish, one with 👏 and a 1-star rating.
**Speaker notes:** "Every day lakhs of people review Blinkit, Zepto and Instamart in Hinglish — Hindi written in English letters mixed with English. We built a system that reads such a review and tells you, per aspect, what the customer liked or disliked, and whether they were being sarcastic."

### Slide 2 — Objective
* Input: one Hinglish review (+ optional star rating)
* Output: aspects mentioned → sentiment per aspect → sarcasm flag → explanation
* Constraints we set ourselves: CPU-only, scikit-learn, explainable, runs live
**Visual:** input → black box → 4 outputs diagram (use the Streamlit screenshot later).
**Notes:** "Four outputs, not one. The last one — *why* — is what makes it usable by a product team."

### Slide 3 — Description: why it matters
* Quick-commerce = 10-minute delivery; feedback must be read fast and at scale
* One star tells you *that* something is wrong, not *what* (delivery? milk? rider? app?)
* Hinglish reviews are ignored by English-only tools
* Sarcastic praise ("great job, 2 ghante late 👏") is counted as *positive* by naive sentiment tools
**Visual:** bar chart "aspects mentioned in our dataset" (`results/fig_dataset.png`).
**Notes:** Mention the scale (each app has crores of installs) and that aspect-level feedback is what operations teams act on.

### Slide 4 — Description: challenges
* No spelling standard: accha / acha / achha / achaa
* Code-mixing: "delivery bahut late thi" — two languages in one clause
* One review, many aspects, opposite sentiments ("fast but expired")
* Sarcasm flips meaning; ~10% of reviews
* Little labelled data (we label our own)
**Visual:** one real review annotated with arrows pointing to each challenge.
**Notes:** "These five challenges map one-to-one to the five main stages of our pipeline."

### Slide 5 — How it is handled today & limitations
* Manual reading by support/ops teams → slow, sampled, inconsistent
* Star-rating dashboards → no aspect, no reason
* Off-the-shelf English sentiment APIs → fail on Hinglish spelling; take sarcasm literally
* Research systems → tweets not reviews, sentence-level only, GPU transformers (SentiMix 2020; ABSA-Mix 2025 is synthetic Devanagari)
**Visual:** 2×2 table: Manual / Rating dashboards / English APIs / Research — columns: Aspect? Hinglish? Sarcasm? Explainable?
**Notes:** Cite the papers verbally (Patwa 2020; Mamta & Ekbal 2025; Shah & Maurya 2021).

### Slide 6 — Use case
* Ops dashboard: % negative on *delivery speed* per city per week
* Product team: top complaints about the *app* this release
* Support triage: sarcastic 1-star reviews auto-routed as complaints
* Demo persona: a Blinkit category manager pastes 500 reviews → gets an aspect heat-map
**Visual:** mock dashboard (aspect × sentiment heat-map).
**Notes:** Emphasise that the per-aspect output is the business value, not an overall score.

### Slide 7 — Proposed solution: pipeline
**Visual (whole slide):** the 8-box pipeline: Normalise → LID + CMI → Aspect extraction → Aspect windows → Aspect sentiment (NB/LR/SVM) → Sarcasm (incongruity) → Polarity correction → Explain (weights + LIME). Mark the three ★ advanced concepts.
**Notes:** Walk left to right in 60 seconds; details on the next three slides.

### Slide 8 — Proposed solution: data
* Source: public Google Play reviews (Blinkit, Zepto, Swiggy) via `google-play-scraper` (MIT) — public data, no login, no user names stored, throttled
* Filter: ≥15% Hindi tokens (our LID) → code-mixed subset; de-duplicated
* Labelled **2,500** reviews (Blinkit 843 / Zepto 836 / Swiggy-Instamart 821) → **3,590** aspect instances: 1,960 neg / 1,485 pos / 145 neu; sarcastic 17 (0.7%)
* Mean Code-Mixing Index 17.1 (only 1.2% of reviews monolingual)
* Public data reused: L3Cube-HingLID (LID training, CC BY-NC-SA), Swami et al. 2018 (sarcasm benchmark)
**Visual:** `results/fig_dataset.png` + `results/fig_cmi.png` side by side.
**Notes:** Say the ethics sentence out loud. From ~8.5K code-mixed reviews (≥10% Hindi tokens) we sampled 2,500 balanced across star ratings. Point out that sarcasm is rare in app reviews — 0.7% — which becomes an honest limitation later.

### Slide 9 — Proposed solution: preprocessing & aspects
* Normalise: lowercase, emoji→words, elongation collapse, 300-entry spelling map, phrase merge
* LID: char n-gram LR trained on L3Cube-HingLID (200K tokens, 4 s) → 96.3% test accuracy, 91.6% on a different corpus (Swami); CMI per review (mean 17.1)
* Aspects: seed lexicon + rules, iterated once on the data (v1 → v2: generic words like *delivery/time/order* replaced by phrases like *late delivery*): micro-F1 vs gold **0.687 → 0.758** (P 0.73 / R 0.79)
* ★ Aspect-conditioned windows: ±5 tokens, cut at *but/lekin/par* and at other aspects
**Visual:** "delivery fast thi | but | doodh expired tha" with two coloured windows.
**Notes:** Explain the window with the example; this is advanced concept 1.

### Slide 10 — Proposed solution: models
* Features: char 2–5-gram TF-IDF + word 1–2-gram TF-IDF + 7 lexicon counts
* Three baselines: ComplementNB · Logistic Regression · Linear SVM (calibrated)
* Imbalance: class-balanced weights; metric: macro-F1
* ★ Sarcasm: 20 sentiment-incongruity features (pos×neg clash, mock-praise + negative, emoji clash, rating-vs-text clash, exaggeration cues) + TF-IDF
* Polarity correction: sarcastic ∧ positive → negative
* Validation: stratified **group** 5-fold CV (a review never splits across folds)
**Visual:** small table of the 20 incongruity features grouped in 4 families.
**Notes:** Advanced concept 2 lives here. Say why char n-grams: spelling noise.

### Slide 11 — Implementation
* Steps: scrape → label sheet → prepare → notebook (train/CV/ablate) → save model → Streamlit demo
* Code: `src/absa/` 8 modules (~1,100 lines), notebook, `app.py`; all CPU; full notebook ≈ 18 min (sarcasm transfer experiments are 10 of those)
* Tools: Python 3, scikit-learn, pandas, emoji, LIME, Streamlit, google-play-scraper, Colab/laptop
* Challenges faced: spelling map coverage, windows for adjacent aspects, calibrated SVM for probabilities, sarcasm label ambiguity, keeping demo data separate from real data
**Visual:** folder tree + 10-line code highlight of `aspect_windows()` or the incongruity features.
**Notes:** Mention the honest-reporting switch (REAL vs DEMO mode) — examiners like it.

### Slide 12 — Results: aspect sentiment
* Table from `results/sentiment_baselines.csv`: ComplementNB 0.869 / 0.713 / 0.719 / **0.716**; LogReg 0.880 / 0.735 / 0.705 / **0.717**; Linear-SVM 0.881 / 0.774 / 0.631 / 0.639 (accuracy / macro-P / macro-R / macro-F1)
* Logistic Regression: accuracy **0.880**, macro-P 0.735, macro-R 0.705, **macro-F1 0.717 ± 0.036** (ComplementNB 0.716, Linear-SVM 0.639 — SVM has high precision but poor neutral recall)
* Per-aspect macro-F1: app/support 0.73, delivery speed 0.73, general 0.72, packaging 0.69, product quality 0.62, price 0.61, delivery partner 0.57 (smallest classes are hardest)
**Visual:** confusion matrix `results/fig_confusion.png` + bar chart of per-aspect F1.
**Notes:** Point at the neutral row of the confusion matrix — neutral is 4% of the data and is the hardest class; that is why we report macro-F1 and not just the 88% accuracy.

### Slide 13 — Results: ablations & sarcasm
* Ablation A (windows): on ALL instances full text 0.735 vs K=5 0.717 (windows cut context on single-topic reviews); on the 2,022 **multi-aspect** instances K=5 **0.612 → 0.653** (+4 pts) — windows help exactly where they are designed to; K=3 too small (0.694), lexicon features worth +2 pts
* Sarcasm on real public benchmark (Swami et al. 2018, 5,250 tweets): LR text-only F1 0.783 → text+incongruity **0.790** (paper: 78.4%)
* Sarcasm on our reviews: only **17 positives / 2,500** → no classifier learns it reliably (best F1 0.17; adding Swami tweets as transfer data does not help — different domain). Honest finding: sarcasm is rare in app reviews
* Ablation B (polarity correction): with predicted flags no change (0.717); with **oracle** flags the sarcastic-subset accuracy jumps **0.684 → 0.895** — the correction rule is right, detection is the bottleneck
**Visual:** two small bar pairs (without/with) — one for windows, one for sarcasm correction.
**Notes:** Be precise and own it: "The detector works on the public benchmark; our review set has too few sarcastic examples to train or test it, so the deployed model is trained on reviews + Swami tweets + 20 hand-written seed examples (training only). The oracle row proves the correction logic."

### Slide 14 — Results: explainability & error analysis
* Explanation example: review → top contributing features (screenshot from app) + LIME word weights
* Error analysis: 431 of 3,590 windows wrong (12%); of the 40 inspected: aspect not found by lexicon 25%, implicit sentiment (no opinion word in window) 22%, neutral↔polar confusion 22%, negation inside window 20%, other 10%
* Real examples: "are also not of good quality" → predicted *positive* (negation scope); "the product price is less compare to other" → *negative* ("less" is implicit praise for price); "need to be more reasonable in price" → *neutral* (polite complaint)
**Visual:** app screenshot of the "Why?" tab; a small pie of error types.
**Notes:** Read one error aloud and say what would fix it (lexicon entry / bigger window / more data).

### Slide 15 — Conclusion
* Findings: 88% accuracy / 0.72 macro-F1 aspect sentiment on CPU in seconds; windows help on multi-aspect reviews (+4 F1) but not on single-topic rants; lexicon iteration gave +7 F1 on aspect extraction; sarcasm is rare (0.7%) in app reviews — detector validated on a public corpus (F1 0.79), correction rule validated with oracle flags
* Scalability: trains in ~6 s, predicts hundreds of reviews/s on a laptop; batch-score an app's reviews nightly; lexicon is a plain Python dict
* Limitations: 2,500 reviews, single-annotator labels, 17 sarcastic examples, lexicon misses 21% of gold aspects, one domain, Roman script only
* Future work: active-learning labelling, HingBERT comparison on GPU, Devanagari support, implicit-aspect detection, multi-platform (X/Reddit)
**Visual:** 3 columns: Findings / Scalability / Future.

### Slide 16 — References (APA-ish)
1. Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T., & Das, A. (2020). SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. *Proc. SemEval-2020*, 774–790.
2. Nayak, R., & Joshi, R. (2022). L3Cube-HingCorpus and HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models. *Proc. WILDRE-6 @ LREC 2022*, 7–12.
3. Mamta, & Ekbal, A. (2025). Quality achhi hai (is good), satisfied! Towards aspect based sentiment analysis in code-mixed language. *Computer Speech & Language*, 89, 101668.
4. Swami, S., Khandelwal, A., Singh, V., Akhtar, S. S., & Shrivastava, M. (2018). A Corpus of English-Hindi Code-Mixed Tweets for Sarcasm Detection. *CICLing 2018*; arXiv:1805.11869.
5. Shah, A., & Maurya, C. (2021). How effective is incongruity? Implications for code-mixed sarcasm detection. *Proc. ICON 2021*, 271–276.
6. Mazumder, D., Kumar, A., & Patro, J. (2024). Revealing the impact of synthetic native samples and multi-tasking strategies in Hindi-English code-mixed humour and sarcasm detection. arXiv:2412.12761.
7. Khanuja, S., et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730.
8. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proc. KDD 2016*.
9. Das, A., & Gambäck, B. (2014). Identifying languages at the word level in code-mixed Indian social media text. *Proc. ICON 2014*. (Code-Mixing Index)
10. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
11. google-play-scraper (Python), https://pypi.org/project/google-play-scraper/ ; LIME, https://github.com/marcotcr/lime

### Slide 17 — Collaboration (keep ONE of the two versions)
**Solo version — "Roles I played":**
* Research & data: literature survey, scraping script, labelling guideline, labelled 2,500 reviews
* Engineering: normaliser, LID/CMI, aspect lexicon + windows, models, Streamlit app
* Evaluation: CV, ablations, error analysis, report & slides
**Team-of-4 version (edit names; shrink to 2–3 rows for smaller teams):**
| Member | Responsibility | Deliverable |
|---|---|---|
| A | Data lead: scraping, filtering, labelling sheet, annotator 1, ethics note | `scripts/`, `data/labelled/` |
| B | NLP preprocessing: normaliser, spelling map, LID + CMI, aspect lexicon & windows, annotator 2 | `normalize.py`, `lid.py`, `aspects.py` |
| C | Modelling & evaluation: baselines, incongruity features, sarcasm, CV, ablations, error analysis | `pipeline.py`, `features.py`, notebook §5–10 |
| D | Explainability, demo & docs: LIME/feature weights, Streamlit app, PPT, report, viva pack | `explain.py`, `app.py`, `docs/` |
All: weekly sync, shared GitHub repo, every member can run the full pipeline and explain every stage.
**Notes:** Examiners often ask "what did *you* do?" — each member should own one stage end-to-end but be able to explain all of them.
