# 3–4 minute live demo script

**Before the teacher arrives (5 min prep):**
1. Terminal 1: `cd hinglish_absa && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
2. Make sure `models/absa_model.joblib` exists (run the notebook once if not).
3. Terminal 1: `streamlit run app.py` → browser opens at http://localhost:8501. Keep the sidebar open.
4. Terminal 2 / second browser tab: the executed notebook (`notebooks/hinglish_absa.ipynb`) scrolled to the results tables; and `results/` folder open.
5. Close everything else. Zoom the browser to 125%.

---
**0:00 – 0:30 — Frame it (say, while the app is on screen)**
"Sir/Ma'am, this is our live system. You paste any Hinglish quick-commerce review; it returns the aspects, the sentiment for each aspect, a sarcasm flag, and the evidence for every decision. Everything runs on this laptop's CPU with scikit-learn — no GPU, no API."

**0:30 – 1:15 — Easy, multi-aspect review (shows aspects + windows)**
Click the sidebar example *"delivery fast thi but doodh expired tha, packing bhi acchi thi"* → **Analyse**.
Say: "Three aspects found. Delivery speed is positive, product quality negative, packaging positive — opposite sentiments in one sentence. The last column shows the **context window** each decision used: the window for delivery stops at *but*, so the word *expired* never reaches the delivery classifier. That is our aspect-conditioned window."
Open **Why? → Sentiment evidence** and point at one row: "For product quality, the biggest contributions are the negative-lexicon count and the word *expired* — exact weights, not a guess."

**1:15 – 2:15 — Sarcastic review (shows incongruity + polarity correction)**
Click *"Wah bhai kya baat hai, doodh expired bhej diya 👏 great job"*, set star rating = 1 → **Analyse**.
Say: "The words are mostly praise — *wah, kya baat, great job*, clapping emoji. A plain sentiment model says positive. Our sarcasm detector says **sarcastic, probability 0.99**. Look at the aspect table: product quality shows *negative — flipped from positive because of sarcasm*. That is the polarity correction."
Open **Why? → Sarcasm evidence**: "The top features are `rating_text_clash` — one star but positive words — and `marker_with_neg` — a praise marker next to a negative word. These are the sentiment-incongruity features."

**2:15 – 2:45 — Type a fresh one (shows robustness to spelling)**
Type: *"Delivry boy bhut rude tha, 40 mint late, par sabji taza thi"* → Analyse.
Say: "Note the spellings *delivry, bhut, mint, sabji* — the normaliser maps them, and the char n-grams handle whatever it misses. Three aspects again: partner negative, speed negative, quality positive." Open the third tab: "Here is the normalised text and the per-word language tags — `hi` / `en` — and the Code-Mixing Index of 40."

**2:45 – 3:30 — Switch to the notebook (shows evaluation is real)**
Scroll to section 5 table: "2,500 real reviews, 3,590 aspect labels. Three baselines, stratified group 5-fold CV; Logistic Regression gives 88% accuracy and macro-F1 0.72." Scroll to the ablation: "Windows versus full text — on multi-aspect reviews the window wins, 0.61 to 0.65. Sarcasm: on the public Swami corpus of 5,250 real tweets our detector gets F1 0.79, matching the original paper. In our reviews only 17 of 2,500 are sarcastic — too few to measure — so we show the oracle: when the flag is right, accuracy on those reviews goes from 0.68 to 0.90."
Scroll to the error table: "And here are the real mistakes with the reason category — missed aspect words, implicit sentiment, neutral confusion and negation, roughly a quarter each."

**3:30 – 3:50 — Close**
"To summarise: aspect-level, Hinglish-aware, sarcasm-corrected, explainable, CPU-only. The pipeline is eight small modules, the whole notebook re-runs in a few minutes, and the lexicons are plain Python dictionaries anyone can extend. Happy to take questions."

---
**If something breaks**
* App not loading → `streamlit run app.py --server.port 8502`.
* Model missing → `jupyter nbconvert --to notebook --execute --inplace notebooks/hinglish_absa.ipynb` (3 min) — talk through the notebook while it runs.
* LIME slow → leave the LIME checkbox off; feature contributions are instant.
* Someone asks for a review you have not tested → type it; if the aspect is missed, say "that word is not in our seed lexicon yet — adding it is a one-line change in `lexicon.py`" and show the file.
