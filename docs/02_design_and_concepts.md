# Step 2 — Project Design and the Concepts Behind It

Written so every paragraph can be lifted into the project report later. Plain language first, then the "why".

## 1. Problem in one sentence
Given a Romanised Hinglish review of a quick-commerce app (Blinkit / Zepto / Instamart), output **which aspects** it talks about, the **sentiment for each aspect**, whether the review is **sarcastic**, and **why** the system decided that.

## 2. Pipeline (and the reason each stage exists)

```
raw review ─► 1 normalise ─► 2 LID + CMI ─► 3 aspect extraction ─► 4 aspect windows
          ─► 5 aspect sentiment (NB / LR / SVM) ─► 6 sarcasm (incongruity) ─► 7 polarity correction ─► 8 explain
```

| Stage | What it does | Why it is needed |
|---|---|---|
| 1 Normalise (`normalize.py`) | lowercase, emoji→words, collapse elongations (`soooo`→`soo`), spelling map (`bkwas`→`bakwas`, `nhi`→`nahi`), phrase merge (`delivery boy`→`delivery_partner`), keep `!`/`?` | Hinglish has no standard spelling; without this the same word is 10 different features. Counts of elongations/`!` are kept as exaggeration cues for sarcasm. |
| 2 LID + CMI (`lid.py`) | char-n-gram LR tags every word `hi`/`en`/`other`; Code-Mixing Index per review | Shows the data really is code-mixed (dataset statistic), gives a feature (mixed reviews are often emotional), and lets us say which lexicon applies to which word. Trained on real public data (L3Cube-HingLID). |
| 3 Aspect extraction (`aspects.py`, `lexicon.py`) | seed lexicon per aspect + ~170 disambiguating phrases (`late delivery`, `food quality`, `extra charges`) + rules (longest phrase first, `general` fallback) | With ~2K labelled reviews, a supervised aspect tagger would be data-starved; a lexicon is transparent, works on day one and we **measure** it against gold labels: v1 micro-F1 0.687 → v2 **0.758** after one data-driven iteration (generic triggers like *delivery/time/order* removed, phrases added). |
| 4 Aspect-conditioned windows | ±K tokens around each trigger, cut at clause breakers and at other aspects' triggers | See §3.1. |
| 5 Aspect sentiment (`pipeline.py`) | char(2–5)+word(1–2) TF-IDF + 7 lexicon counts → NB / LR / Linear-SVM, class-balanced, stratified group 5-fold CV | Strong, fast, explainable baselines; char n-grams absorb remaining spelling noise. |
| 6 Sarcasm (`features.py`) | TF-IDF + 20 **sentiment-incongruity** features → same three classifiers | See §3.2. |
| 7 Polarity correction | if sarcastic and aspect sentiment = positive → negative | Sarcastic reviews are mock praise; the literal positive is wrong. Neutral/negative untouched. |
| 8 Explain (`explain.py`) | exact per-feature contributions (w·x) + LIME | See §3.3. |

## 3. The three advanced concepts, explained for the viva

### 3.1 Aspect-conditioned context windows
*Problem:* "delivery fast thi **but** doodh expired tha" has opposite sentiments for two aspects. A classifier that reads the whole review must output one label — it cannot be right for both.
*Idea:* sentiment words sit close to the aspect they describe. For each aspect mention we keep only the tokens within ±K positions (K = 5 by default) and we stop the window at a **clause breaker** (`but`, `lekin`, `par`, `magar`, `!`) or at a trigger of a **different aspect**. The classifier then sees `delivery fast thi` for delivery_speed and `doodh expired tha` for product_quality. If an aspect is mentioned twice, the windows are merged (union of positions) so nothing is duplicated.
*Why it is "conditioned":* the input to the classifier depends on (is conditioned on) which aspect we are asking about — the same review gives different inputs for different aspects.
*Evidence (real data, 2,500 reviews):* `results/ablation_windows.csv` — on ALL 3,590 instances full text scores macro-F1 0.735 vs K=5 window 0.717 (most Play Store reviews are single-topic, so cutting context only loses words); on the 2,022 **multi-aspect** instances (`results/ablation_windows_multiaspect.csv`) the window wins **0.612 → 0.653**. K=3 is too small (0.694), K=8 ≈ K=5 (0.720), and removing the clause stop costs ~1 point. Report both tables: the method helps exactly where it is designed to and not elsewhere.
*Limitation to admit:* long-distance references ("the milk… it was expired") and sentiment words outside the window are missed; K is a trade-off (too small → no sentiment word inside; too large → neighbouring aspect leaks in).

### 3.2 Sentiment-incongruity-based sarcasm detection
*Problem:* "Wah bhai kya baat hai, doodh expired bhej diya 👏 great job" contains more positive than negative words; a bag-of-words sentiment model says *positive*.
*Idea (incongruity):* sarcasm is a **clash** between what the words say and what the situation is. We make the clash measurable with 20 features, e.g. `incongruity_flag` (both positive and negative lexicon words present), `incongruity_strength` (strongest positive × strongest negative), `marker_with_neg` (mock-praise marker such as *wah / great job / 👏* **and** a negative word), `emoji_text_clash`, `rating_text_clash` (1–2 stars but positive words dominate — a signal tweets never have), elongation / `!` / ALL-CAPS counts (exaggeration), CMI. These are concatenated with TF-IDF and fed to a class-balanced classifier.
*Why features and not a neural self-matching network (Shah & Maurya 2021):* with ~2K reviews and a CPU, explicit features are data-efficient, trainable in seconds and each one can be shown in the demo ("the review is sarcastic mainly because of `rating_text_clash` and `marker_with_neg`").
*Evidence:* `results/sarcasm_swami_benchmark.csv` — on the real Swami et al. corpus (5,250 tweets, 10% sarcastic, no ratings) LR text-only F1 = 0.783 vs text+incongruity **0.790** (precision 0.68, recall 0.95); the original paper's hand-crafted-feature system reached 78.4%. On our reviews the honest finding is that sarcasm is **rare: 17 of 2,500 (0.7%)** — too few to train or evaluate (best CV F1 0.17; adding the Swami tweets to the training folds does not help because of domain shift). Ablation B (`results/ablation_sarcasm.csv`): predicted flags change nothing, but **oracle** flags raise accuracy on the sarcastic-review instances from 0.684 to 0.895 — the correction rule is validated, detection is the bottleneck. The deployed model is therefore trained on reviews + Swami + 20 hand-written seed examples (training-only augmentation, documented in the notebook).
*Limitation to admit:* sarcasm with no lexicon clash ("classic Blinkit, as usual") is missed; hashtag/distant labels in public data are noisy; annotators disagree on sarcasm; false positives flip genuinely positive reviews to negative, so the threshold matters.

### 3.3 Explainability: feature-weight inspection and LIME
*Exact method (linear models):* for LR/Linear-SVM the class score is b + Σ wᵢ·xᵢ, so each feature's contribution is simply wᵢ·xᵢ for the actual input. We rank these and show the top ones (e.g. `lexicon: win_neg_count +1.28`, `word: expired +0.10`). No approximation, instant. We also show the **global** top-weighted features per class (what the model learned overall) — a quick sanity check that it learned sentiment words, not artefacts.
*LIME (model-agnostic):* LIME perturbs the text by dropping words, queries the model hundreds of times, and fits a local linear model; it reports word-level weights and works for any classifier (NB, transformers). Slower (~2–5 s on CPU with 300 samples), so it is a checkbox in the app.
*Why both:* exact weights are the honest explanation for our linear models; LIME is easier to read for non-technical users and would still work if the model were swapped for a transformer.

## 4. Model and evaluation choices
**Headline results (real data, 5-fold stratified group CV):** Logistic Regression accuracy 0.880, macro-P 0.735, macro-R 0.705, macro-F1 0.717 ± 0.036; ComplementNB 0.716; Linear-SVM 0.639. Per-aspect macro-F1: app/support 0.73, delivery speed 0.73, general 0.72, packaging 0.69, product quality 0.62, price 0.61, delivery partner 0.57.

* **Three baselines, not one:** ComplementNB (strong with little data + imbalance), Logistic Regression (probabilities + interpretable weights), Linear SVM (often best for text; wrapped in `CalibratedClassifierCV` to get probabilities for LIME).
* **Features:** char_wb 2–5-grams (handle spelling variation), word 1–2-grams (phrases like `bahut late`), 7 window lexicon counts (positive/negative/negator/intensifier counts, max scores, length). Sub-linear TF.
* **Imbalance:** `class_weight="balanced"` (LR, SVM) / balanced `sample_weight` (NB); we report **macro-F1** (treats neutral and sarcastic classes as equally important) plus per-class precision/recall.
* **Cross-validation:** `StratifiedGroupKFold(5)` with the review as the group → all windows of one review stay in one fold (no leakage), class ratios preserved. Sarcasm: `StratifiedKFold(5)` on reviews.
* **Ablations:** A) windows vs full text, K, clause stop, lexicon features; B) with vs without sarcasm correction (+ oracle); C) sarcasm feature sets (text / incongruity / both) × three classifiers.
* **Error analysis:** 431 of 3,590 out-of-fold windows are wrong (12%). The first 40 are dumped to `results/errors_sentiment.csv` with an automatic error type: aspect not found by lexicon 25%, implicit sentiment (no opinion word in the window) 22%, neutral↔polar 22%, negation inside the window 20%, other 10%.
* **Optional transformer:** notebook §12 (`RUN_TRANSFORMER=True`) adds a frozen multilingual sentence-embedding + LR row on the same folds — a fair, CPU-feasible comparison. Full MuRIL/XLM-R fine-tuning is left out: it needs a ~1 GB download, 30–60 min/fold on CPU, and published code-mixed results (SentiMix, Mazumder et al.) show its advantage comes with GPUs and far more data than a lab project has.

## 5. Why not deep learning as the core? (one-liners you can say)
* "Our labelled set is ~2K reviews; char-n-gram linear models reach their best with that much data, deep models do not."
* "The constraint is CPU-only; everything here trains in seconds, so we could run 20 ablation configurations in one notebook."
* "Linear models give exact explanations, which the demo needs."
* "SemEval-2020 SentiMix showed BERT-style models win on 20K tweets, but with a modest margin over strong linear baselines — and we kept a transformer row as an optional comparison to show we know the trade-off."
* "The hard parts of this problem (which aspect, is it sarcastic) are about *domain structure*, which we encode explicitly; a bigger model would not know that `handling fee` is a price aspect without the same labelled data."
