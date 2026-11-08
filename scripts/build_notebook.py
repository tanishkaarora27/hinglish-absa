"""Generate notebooks/hinglish_absa.ipynb (run once; the notebook is committed)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Aspect-Based Sentiment Analysis of Hinglish Quick-Commerce Reviews with Sarcasm Detection

**Pipeline:** normalise → word-level language ID + Code-Mixing Index → aspect extraction (seed lexicon + rules) → aspect-conditioned context windows → aspect sentiment (NB / LR / Linear-SVM on char n-gram TF-IDF) → sarcasm detection (sentiment-incongruity features) → polarity correction → explanation (feature weights + LIME).

Runs on CPU (laptop or free Colab). No GPU, no paid service.

### Which data does this notebook use?
* **REAL mode** – if `data/labelled/reviews.csv` and `data/labelled/aspect_labels.csv` exist (produced from your own scraped + hand-labelled Blinkit / Zepto / Instamart reviews via `scripts/prepare_labelled_data.py`), everything below is computed on them and the numbers are reportable.
* **DEMO mode** – otherwise it falls back to `data/demo/` (149 **hand-written seed reviews** that exist to prove the code runs). **Numbers from DEMO mode must NOT be put in the report / PPT.** Every output in this mode is prefixed with `[DEMO-SYNTHETIC]`.

In REAL mode the seed reviews are used for exactly one thing: 20 of them are sarcastic, and sarcasm is so rare in real Play Store reviews (<1%) that the *deployed* sarcasm detector (section 11) is trained on real reviews + the public Swami corpus + these seed examples. They are **never** used for any evaluation number.

Two experiments always use **real public data** regardless of mode: (1) the language-ID model is trained/evaluated on L3Cube-HingLID and cross-checked on Swami et al. (2018); (2) the sarcasm detector is benchmarked on the Swami et al. (2018) sarcasm corpus (5,250 real Hinglish tweets). Run `python scripts/download_external_data.py --core-only` first.""")

code(r"""# ---- 0. setup -------------------------------------------------------------
import sys, os, json, time, warnings, pathlib
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd()
if not (ROOT / "src").exists():          # notebook started from notebooks/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 90)

from absa.normalize import normalize
from absa.lid import LanguageIdentifier, read_l3cube_lid, read_swami_lid
from absa.aspects import ASPECTS, aspect_windows, detect_aspects, extract_mentions
from absa.pipeline import HinglishABSA, build_sentiment_model, build_sarcasm_model, fit_with_balance
from absa import evaluation as ev, explain as ex
from absa.features import incongruity_report

RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
EXT = ROOT / "data" / "external"
SEED = 42; np.random.seed(SEED)
print("project root:", ROOT)""")

md("## 1. Load the review dataset (REAL or DEMO)")
code(r"""lab_dir = ROOT / "data" / "labelled"
if (lab_dir / "reviews.csv").exists() and (lab_dir / "aspect_labels.csv").exists():
    DATA_MODE = "REAL"; data_dir = lab_dir
else:
    DATA_MODE = "DEMO-SYNTHETIC"; data_dir = ROOT / "data" / "demo"
TAG = f"[{DATA_MODE}]"
reviews = pd.read_csv(data_dir / "reviews.csv")
labels  = pd.read_csv(data_dir / "aspect_labels.csv")
labels = labels[labels["aspect"].isin(ASPECTS)].reset_index(drop=True)
reviews["rating"] = pd.to_numeric(reviews.get("rating", np.nan), errors="coerce")
reviews["sarcastic"] = reviews.get("sarcastic", 0).fillna(0).astype(int)
print(TAG, "reviews:", len(reviews), "| aspect instances:", len(labels),
      "| sarcastic reviews:", int(reviews.sarcastic.sum()), f"({reviews.sarcastic.mean():.1%})")
if DATA_MODE != "REAL":
    print("\n*** DEMO-SYNTHETIC MODE: the reviews below were written by hand to test the code. "
          "Do NOT report these metrics. Add data/labelled/ to switch to REAL mode. ***\n")
display(reviews.head())
stats = {"mode": DATA_MODE, "n_reviews": int(len(reviews)), "n_instances": int(len(labels)),
         "sarcastic_rate": float(reviews.sarcastic.mean()),
         "sentiment_dist": labels.sentiment.value_counts().to_dict(),
         "aspect_dist": labels.aspect.value_counts().to_dict()}
display(pd.crosstab(labels.aspect, labels.sentiment, margins=True))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
labels.aspect.value_counts().plot.barh(ax=ax[0], title=f"{TAG} aspect instances")
labels.sentiment.value_counts().plot.bar(ax=ax[1], title=f"{TAG} sentiment labels", rot=0)
plt.tight_layout(); plt.savefig(RESULTS / "fig_dataset.png", dpi=130); plt.show()""")

md("""## 2. Stage 1 — Text normalisation
Hinglish spelling is free-form. We lowercase, demojise, collapse elongations, map known spelling variants and keep `!`/`?` as tokens. The cell also lists the most frequent tokens that are **not** in our spelling map and not in any lexicon — look at that list to extend `normalize.SPELLING_MAP` for your own data.""")
code(r"""examples = ["Delivery sooooo late thi, 10 min bola 1 ghanta laga 😡 bkwas service!!",
            "Wah bhai kya baat hai, doodh expired bhej diya 👏 great job",
            "Delivry boy bhut polite tha, pkging bhi acchi thi 👍"]
for s in examples:
    n = normalize(s)
    print(f"RAW : {s}\nNORM: {n.text}\n      elongations={n.n_elongations} exclaim={n.n_exclaim} emoji={n.emojis}\n")

from collections import Counter
from absa import lexicon as LX
known = set(LX.POSITIVE) | set(LX.NEGATIVE) | set(LX.NEGATORS) | set(LX.INTENSIFIERS) | \
        {t for v in LX.ASPECT_LEXICON.values() for t in v} | set(LX.SARCASM_MARKERS)
STOP = set("to the and i is it a for me in they but this my you was are with have from that on when or also your all be get more if of as at by so not no we he she his her them their there here what which who how do does did can could would should will just very only even than then too one any some about out up now its im am been being has had our us ko ka ki ke se ne mein par pe bhi hi nahi na ho hain hoon tha thi ye yeh wo woh koi kuch kar karo karte karta karti kiya kya kyu aur ya ab abhi phir fir jab tab jo jis hota hoti hote raha rahi rahe diya di de do le lo liya li mil mila milta aa aaya aati aata gaya gayi gaye bahut sab app aap apne apna apni mere mera meri hum mujhe main mai tum ek use".split())
cnt = Counter(t for txt in reviews.text for t in normalize(txt).tokens
              if t not in known and t not in STOP and t.isalpha() and len(t) > 2)
print(TAG, "most frequent tokens outside all lexicons (candidates for the spelling map / lexicon):")
print(cnt.most_common(40))""")

md("""## 3. Stage 2 — Word-level language identification + Code-Mixing Index
A char n-gram (1–4) TF-IDF + logistic regression classifier trained on **L3Cube-HingLID** (real Hinglish tweets, CC BY-NC-SA 4.0). We evaluate on its held-out test split and *cross-dataset* on the word-level tags of **Swami et al. (2018)**. Then we compute the CMI of every review in our dataset.""")
code(r"""t0 = time.time()
w_tr, l_tr = read_l3cube_lid(str(EXT / "l3cube_hinglid_train.txt"), max_tokens=200_000)
lid = LanguageIdentifier().fit(w_tr, l_tr)
w_te, l_te = read_l3cube_lid(str(EXT / "l3cube_hinglid_test.txt"))
acc_l3 = float(np.mean(lid.model.predict(w_te) == np.array(l_te)))
w_sw, l_sw = read_swami_lid(str(EXT / "swami_lid_tags.txt"))
acc_sw = float(np.mean(lid.model.predict(w_sw) == np.array(l_sw)))
print(f"[REAL PUBLIC DATA] LID trained on {len(w_tr):,} tokens in {time.time()-t0:.1f}s")
print(f"  L3Cube-HingLID test accuracy  : {acc_l3:.4f}  (n={len(w_te):,} tokens)")
print(f"  Swami et al. cross-dataset acc: {acc_sw:.4f}  (n={len(w_sw):,} tokens)")
lid.save(MODELS / "lid.joblib")
lid_metrics = {"l3cube_test_acc": acc_l3, "swami_cross_acc": acc_sw, "train_tokens": len(w_tr)}

# CMI of our reviews
cmis = [lid.analyse(normalize(t).tokens).cmi for t in reviews.text]
reviews["cmi"] = cmis
print(f"\n{TAG} CMI: mean={np.mean(cmis):.1f}, median={np.median(cmis):.1f}, "
      f"monolingual (CMI=0): {np.mean(np.array(cmis)==0):.1%}")
plt.figure(figsize=(6,3)); plt.hist(cmis, bins=20); plt.title(f"{TAG} Code-Mixing Index per review")
plt.xlabel("CMI (0 = monolingual, 50 = perfectly mixed)"); plt.tight_layout()
plt.savefig(RESULTS / "fig_cmi.png", dpi=130); plt.show()
n = normalize(examples[0]); r = lid.analyse(n.tokens)
print(list(zip(r.tokens, r.langs)), "CMI =", r.cmi)""")

md("""## 4. Stage 3 — Aspect extraction (seed lexicon + rules) and context windows
We check the rule-based extractor against the gold aspects (multi-label precision / recall / F1 per aspect) and show the aspect-conditioned windows.""")
code(r"""asp_rep = ev.aspect_extraction_report(reviews, labels)
print(TAG, "aspect extraction vs gold labels"); display(asp_rep)
asp_rep.to_csv(RESULTS / "aspect_extraction.csv", index=False)
for s in ["delivery fast thi but doodh expired tha, packing bhi thik thi",
          "Price thoda mehnga hai lekin delivery boy bahut polite tha aur app smooth chalta hai"]:
    toks = normalize(s).tokens
    print("\n", s); print("  mentions:", [(m.aspect, m.trigger) for m in extract_mentions(toks)])
    for k in (3, 5):
        print(f"  windows k={k}:", aspect_windows(toks, k=k))
    print("  no-window :", aspect_windows(toks, use_window=False))""")

md("""## 5. Stage 4 — Aspect-conditioned sentiment classification: three baselines
For every gold (review, aspect) pair we build the window text and train three classifiers on char(2–5)+word(1–2) TF-IDF + 7 window-lexicon counts. Class imbalance → `class_weight='balanced'` (LR, SVM) or balanced `sample_weight` (NB). Evaluation → **stratified group 5-fold CV** (all windows of one review stay in one fold). Macro-F1 is the headline metric because the classes are imbalanced.""")
code(r"""system = HinglishABSA(lid=lid, window_k=5)
inst = system.build_instances(reviews, labels)
print(TAG, f"{len(inst)} instances; aspect found by lexicon for {inst.found.mean():.1%} (others use full text)")
sent_results, oof = {}, {}
for kind in ["nb", "lr", "svm"]:
    t0 = time.time()
    res = ev.cv_sentiment(inst, lambda k=kind: build_sentiment_model(k), kind, seed=SEED)
    oof[kind] = res.pop("oof_pred"); res["time_s"] = round(time.time() - t0, 1)
    sent_results[kind] = res
tbl = pd.DataFrame(sent_results).T.rename(index={"nb": "ComplementNB", "lr": "LogisticRegression", "svm": "LinearSVM(calibrated)"})
print(TAG, "aspect sentiment — 5-fold stratified group CV"); display(tbl)
tbl.to_csv(RESULTS / "sentiment_baselines.csv")
BEST = max(sent_results, key=lambda k: sent_results[k]["f1_macro"])
print("best by macro-F1:", BEST)""")

code(r"""print(TAG, f"per-aspect results ({BEST})"); per_aspect = ev.per_aspect_table(inst, oof[BEST]); display(per_aspect)
per_aspect.to_csv(RESULTS / "per_aspect.csv", index=False)
cm = ev.confusion(inst.sentiment, oof[BEST]); display(cm)
print(ev.report_text(inst.sentiment, oof[BEST]))
fig, ax = plt.subplots(figsize=(4.5, 4)); im = ax.imshow(cm.values, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(ev.SENTIMENT_ORDER); ax.set_yticklabels(ev.SENTIMENT_ORDER)
for i in range(3):
    for j in range(3): ax.text(j, i, cm.values[i, j], ha="center", va="center", color="k")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{TAG} confusion ({BEST})")
plt.tight_layout(); plt.savefig(RESULTS / "fig_confusion.png", dpi=130); plt.show()""")

md("""## 6. Ablation A — do aspect windows help?
Same classifier, different inputs: full review text (no window) vs windows of size 3 / 5 / 8, with and without stopping at clause breakers, and with/without the 7 lexicon features.""")
code(r"""abl_rows = []
configs = [("no window (full text)", dict(use_window=False)),
           ("window k=3", dict(window_k=3)), ("window k=5", dict(window_k=5)), ("window k=8", dict(window_k=8)),
           ("window k=5, no clause stop", dict(window_k=5, stop_at_clause=False)),
           ("window k=5, no lexicon feats", dict(window_k=5, use_lexicon_feats=False))]
for name, kw in configs:
    sysm = HinglishABSA(lid=lid, **{k: v for k, v in kw.items() if k != "use_lexicon_feats"})
    ins = sysm.build_instances(reviews, labels)
    ulf = kw.get("use_lexicon_feats", True)
    res = ev.cv_sentiment(ins, lambda k=BEST, u=ulf: build_sentiment_model(k, use_lexicon_feats=u), BEST, seed=SEED)
    abl_rows.append({"setting": name, "f1_macro": res["f1_macro"], "±std": res["f1_macro_std"],
                     "accuracy": res["accuracy"], "precision_macro": res["precision_macro"], "recall_macro": res["recall_macro"]})
ablation_windows = pd.DataFrame(abl_rows); print(TAG, f"window ablation ({BEST})"); display(ablation_windows)
ablation_windows.to_csv(RESULTS / "ablation_windows.csv", index=False)

# Windows can only matter where a review carries MORE THAN ONE aspect (otherwise window ~= full text).
multi = labels.groupby("review_id").aspect.nunique(); multi_ids = set(multi[multi >= 2].index)
rows = []
for name, kw in [("no window (full text)", dict(use_window=False)), ("window k=5", dict(window_k=5))]:
    sysm = HinglishABSA(lid=lid, **kw); ins = sysm.build_instances(reviews, labels)
    res = ev.cv_sentiment(ins, lambda k=BEST: build_sentiment_model(k), BEST, seed=SEED)
    m = ins.review_id.isin(multi_ids).to_numpy()
    sub = ev.summarise(ins.sentiment[m], res["oof_pred"][m], labels=ev.SENTIMENT_ORDER) if m.any() else {}
    rows.append({"setting": name, "multi_aspect_instances": int(m.sum()), **{f"multi_{k}": v for k, v in sub.items()}})
ablation_multi = pd.DataFrame(rows); print(TAG, "same ablation restricted to multi-aspect reviews"); display(ablation_multi)
ablation_multi.to_csv(RESULTS / "ablation_windows_multiaspect.csv", index=False)""")

md("""## 7. Stage 5 — Sarcasm detection with sentiment-incongruity features
**7a. Public benchmark (real data).** Swami et al. (2018): 5,250 Hinglish tweets, 504 sarcastic (~10%). We compare text-only TF-IDF, incongruity-only, and both, for NB / LR / SVM with 5-fold stratified CV. The original paper reports 78.4% F1 with hand-crafted features + random forest.

**7b. Our review dataset.** Same models on the reviews (here the `rating_text_clash` feature is available, which tweets do not have).""")
code(r"""swami = pd.read_csv(EXT / "swami_sarcasm.csv")
print(f"[REAL PUBLIC DATA] Swami et al. 2018: {len(swami)} tweets, {swami.sarcastic.sum()} sarcastic")
rows = []
for kind in ["nb", "lr", "svm"]:
    for fname, kw in [("text only", dict(use_incongruity=False)), ("incongruity only", dict(use_text=False)), ("text + incongruity", {})]:
        t0 = time.time()
        res = ev.cv_sarcasm(swami, lambda k=kind, kw=kw: build_sarcasm_model(k, lid=lid, **kw), kind, seed=SEED)
        rows.append({"model": kind, "features": fname, "f1_sarcastic": res["f1_sarcastic"], "precision_sarcastic": res["precision_sarcastic"],
                     "recall_sarcastic": res["recall_sarcastic"], "f1_macro": res["f1_macro"], "accuracy": res["accuracy"], "time_s": round(time.time()-t0, 1)})
swami_tbl = pd.DataFrame(rows); print("[REAL PUBLIC DATA] sarcasm on Swami et al. (5-fold CV)"); display(swami_tbl)
swami_tbl.to_csv(RESULTS / "sarcasm_swami_benchmark.csv", index=False)""")

code(r"""rows, sarc_oof = [], {}
n_sarc = int(reviews.sarcastic.sum())
print(TAG, f"{n_sarc} sarcastic reviews out of {len(reviews)} ({n_sarc/len(reviews):.1%})")
if n_sarc >= 5:
    feats = [("text only", dict(use_incongruity=False)), ("incongruity only", dict(use_text=False)), ("text + incongruity", {})]
    # (a) reviews only;  (b) TRANSFER: Swami tweets added to every training fold (tested on reviews only)
    for train_set, extra in [("reviews only", None), ("reviews + Swami (transfer)", swami)]:
        for kind in (["nb", "lr", "svm"] if extra is None else ["nb", "lr"]):
            for fname, kw in feats:
                res = ev.cv_sarcasm(reviews, lambda k=kind, kw=kw: build_sarcasm_model(k, lid=lid, **kw), kind, seed=SEED, extra_train=extra)
                sarc_oof[(train_set, kind, fname)] = res["oof_pred"]
                rows.append({"train_set": train_set, "model": kind, "features": fname, "f1_sarcastic": res["f1_sarcastic"],
                             "precision_sarcastic": res["precision_sarcastic"], "recall_sarcastic": res["recall_sarcastic"],
                             "f1_macro": res["f1_macro"], "accuracy": res["accuracy"]})
    sarc_tbl = pd.DataFrame(rows); print(TAG, "sarcasm on the review dataset (5-fold CV, tested on reviews only)"); display(sarc_tbl)
    sarc_tbl.to_csv(RESULTS / "sarcasm_reviews.csv", index=False)
    SARC_BEST = sarc_tbl.sort_values(["f1_sarcastic", "f1_macro"], ascending=False).iloc[0]
    print("best:", SARC_BEST.train_set, "/", SARC_BEST.model, "/", SARC_BEST.features)
    if n_sarc < 50:
        print(f"\nNOTE: only {n_sarc} positive examples -> these review-level sarcasm numbers are unstable; "
              "the public Swami benchmark above is the reliable measure of the detector. See the limitations section.")
else:
    print(TAG, "too few sarcastic reviews to cross-validate a sarcasm model"); SARC_BEST = None""")

md("""## 8. Ablation B — polarity correction: with vs without sarcasm handling
End-to-end: the out-of-fold aspect-sentiment predictions are corrected with the out-of-fold sarcasm predictions (positive → negative when the review is predicted sarcastic). We measure on all instances and on the sarcastic subset.""")
code(r"""if SARC_BEST is not None:
    sarc_pred = dict(zip(reviews.review_id, sarc_oof[(SARC_BEST.train_set, SARC_BEST.model, SARC_BEST.features)]))
    base = list(oof[BEST])
    corrected = ev.apply_correction(base, sarc_pred, inst.review_id.tolist())
    gold_sarc = dict(zip(reviews.review_id, reviews.sarcastic))
    mask = inst.review_id.map(gold_sarc).astype(bool).to_numpy()
    rows = []
    for name, pred in [("without sarcasm handling", base), ("with sarcasm handling (predicted)", corrected),
                       ("with sarcasm handling (oracle gold flags)", ev.apply_correction(base, gold_sarc, inst.review_id.tolist()))]:
        all_m = ev.summarise(inst.sentiment, pred, labels=ev.SENTIMENT_ORDER)
        sub_m = ev.summarise(inst.sentiment[mask], np.array(pred)[mask], labels=ev.SENTIMENT_ORDER) if mask.any() else {}
        rows.append({"setting": name, "all_f1_macro": all_m["f1_macro"], "all_accuracy": all_m["accuracy"],
                     "sarcastic_subset_accuracy": sub_m.get("accuracy"), "sarcastic_subset_n": int(mask.sum())})
    ablation_sarcasm = pd.DataFrame(rows); print(TAG, "polarity-correction ablation"); display(ablation_sarcasm)
    ablation_sarcasm.to_csv(RESULTS / "ablation_sarcasm.csv", index=False)""")

md("""## 9. Explainability — feature-weight inspection and LIME
* **Global:** the strongest n-grams / lexicon features per class (what did the model learn?).
* **Local, exact:** per-feature contribution `weight × value` for one input (linear model).
* **Local, model-agnostic:** LIME word-level explanation.""")
code(r"""# Deployed model. Aspect sentiment: trained on the review data only.
# Sarcasm: real reviews + public Swami corpus (5,250 tweets) + 149 hand-written seed examples
# (data/demo, 20 sarcastic) -- the seed examples are TRAINING-ONLY augmentation for a <1% class
# and never enter any evaluation number above.
seed_df = pd.read_csv(ROOT / "data" / "demo" / "reviews.csv")[["text", "rating", "sarcastic"]]
extra_sarc = pd.concat([swami[["text", "sarcastic"]], seed_df], ignore_index=True) if DATA_MODE == "REAL" else None
final = HinglishABSA(lid=lid, window_k=5, sentiment_kind=BEST, sarcasm_kind="lr").fit(reviews, labels, extra_sarcasm=extra_sarc)
print("sarcasm model trained on", final.meta.get("sarcasm_extra_train", 0), "extra examples +", len(reviews), "reviews")
print(TAG, "top features per class (global):"); display(ex.top_features(final.sentiment_model, n=8))
ex.top_features(final.sentiment_model, n=15).to_csv(RESULTS / "top_features_sentiment.csv", index=False)
demo_text = "Wah bhai kya baat hai, doodh expired bhej diya 👏 great job"
r = final.predict_review(demo_text, rating=1)
print("\nINPUT:", demo_text); print("sarcastic:", r.sarcastic, f"(p={r.sarcasm_proba:.2f})  CMI={r.cmi}")
for a in r.aspects:
    print(f"  {a.aspect:16s} window='{a.window}'  raw={a.sentiment_raw}  final={a.sentiment_final}  corrected={a.corrected}")
print("\nexact contributions for the window (towards predicted class):")
display(ex.contributions(final.sentiment_model, r.aspects[0].window, n=8))
if final.sarcasm_model is not None:
    print("\nsarcasm model contributions (towards 'sarcastic'):")
    display(ex.contributions(final.sarcasm_model, pd.DataFrame([{"text": demo_text, "rating": 1}]), target_class=1, n=8))
pred, lime_list = ex.lime_explain(final.sentiment_model, r.aspects[0].window, num_samples=300)
print("\nLIME (word level) for predicted class", pred, ":", lime_list)""")

md("""## 10. Error analysis — real misclassified examples
Out-of-fold errors of the best sentiment model, plus sarcasm errors. The last column gives an automatic first guess at the error type; read each row and refine the category by hand for the report.""")
code(r"""err = ev.error_table(inst, oof[BEST], n=40)
def guess(row):
    toks = row.window.split()
    if not row.found: return "aspect not found by lexicon → full text used"
    if any(t in LX.NEGATORS for t in toks): return "negation inside window"
    if row.sentiment == "neutral" or row.pred == "neutral": return "neutral vs polar confusion"
    if not any(t in LX.POSITIVE or t in LX.NEGATIVE for t in toks): return "no sentiment word in window (implicit sentiment)"
    return "mixed / other"
err["error_type"] = err.apply(guess, axis=1)
print(TAG, f"{len(err)} shown of {int((inst.sentiment != oof[BEST]).sum())} sentiment errors")
display(err[["aspect", "sentiment", "pred", "window", "error_type"]])
err.to_csv(RESULTS / "errors_sentiment.csv", index=False)
print(err.error_type.value_counts())
if SARC_BEST is not None:
    se = reviews.assign(pred=sarc_oof[(SARC_BEST.train_set, SARC_BEST.model, SARC_BEST.features)])
    se = se[se.sarcastic != se.pred][["review_id", "text", "rating", "sarcastic", "pred"]]
    print(f"\n{TAG} sarcasm errors ({len(se)}):"); display(se.head(20)); se.to_csv(RESULTS / "errors_sarcasm.csv", index=False)""")

md("## 11. Save the final models + a metrics summary (used by the Streamlit app and the report)")
code(r"""final.meta.update({"data_mode": DATA_MODE, "best_sentiment_model": BEST,
                   "best_sarcasm_model": "lr/text+incongruity (reviews + Swami + seed examples)",
                   "best_sarcasm_cv": None if SARC_BEST is None else f"{SARC_BEST.train_set}/{SARC_BEST.model}/{SARC_BEST.features}",
                   "trained_on": str(data_dir.name)})
final.save(MODELS / "absa_model.joblib")
summary = {"data_mode": DATA_MODE, "dataset": stats, "lid": lid_metrics, "sentiment_cv": sent_results, "best_sentiment": BEST,
           "window_ablation": ablation_windows.to_dict("records"), "swami_sarcasm_benchmark": swami_tbl.to_dict("records"),
           "sarcasm_reviews": (sarc_tbl.to_dict("records") if SARC_BEST is not None else None),
           "polarity_correction_ablation": (ablation_sarcasm.to_dict("records") if SARC_BEST is not None else None)}
(RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
print("saved models/absa_model.joblib, models/lid.joblib, results/metrics.json")
if DATA_MODE != "REAL":
    print("\n*** Reminder: this run used SYNTHETIC demo data. Only the L3Cube-LID and Swami-sarcasm numbers are real. ***")""")

md("""## 12. (Optional) Transformer comparison — why it is optional and when to run it
A pretrained multilingual transformer (e.g. `google/muril-base-cased` or `xlm-roberta-base`) can be fine-tuned on the same windows. **We keep it optional because:** (1) the project constraint is CPU-only and 1,500–3,000 labelled windows — a 110M-parameter model fine-tuned on this little data on CPU takes 30–60 min per fold and typically gains little over char n-gram linear models (SentiMix 2020 top systems used BERT variants but the margin over strong linear baselines was modest); (2) linear models give exact explanations; (3) it needs an internet download (~1 GB). Run the cell below on Colab only if you want the comparison for the report; it reuses the same CV folds so the numbers are comparable. Leave `RUN_TRANSFORMER=False` otherwise.""")
code(r"""RUN_TRANSFORMER = False   # set True on Colab (CPU ok but slow; GPU faster) to add the comparison row
if RUN_TRANSFORMER:
    # pip install transformers torch sentence-transformers  (not in requirements.txt on purpose)
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    # Cheap variant: frozen multilingual embeddings + LR (no fine-tuning, runs on CPU in a few minutes)
    st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Xe = st.encode(inst.window.tolist(), batch_size=32, show_progress_bar=False)
    y = inst.sentiment.to_numpy(); g = inst.review_id.to_numpy()
    oof_t = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xe, y, g):
        oof_t[te] = LogisticRegression(class_weight="balanced", max_iter=3000).fit(Xe[tr], y[tr]).predict(Xe[te])
    print(TAG, "frozen multilingual embeddings + LR:", ev.summarise(y, oof_t, labels=ev.SENTIMENT_ORDER))
else:
    print("transformer comparison skipped (RUN_TRANSFORMER=False)")""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "hinglish_absa.ipynb"
nbf.write(nb, out)
print("wrote", out)
