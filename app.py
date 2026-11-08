"""
Live demo: type a Hinglish quick-commerce review -> aspects, sentiment per
aspect, sarcasm flag, and WHY (feature contributions + optional LIME).

Run:   streamlit run app.py
Needs: models/absa_model.joblib and models/lid.joblib (created by the notebook).
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from absa import explain as ex  # noqa: E402
from absa.pipeline import HinglishABSA  # noqa: E402

st.set_page_config(page_title="Hinglish ABSA + Sarcasm", page_icon="🛒", layout="wide")

ASPECT_LABEL = {"delivery_speed": "Delivery speed", "product_quality": "Product quality / freshness",
                "packaging": "Packaging", "price": "Price & fees", "delivery_partner": "Delivery partner",
                "app_support": "App / customer support", "general": "Overall (no specific aspect)"}
EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}


@st.cache_resource
def load_model():
    path = ROOT / "models" / "absa_model.joblib"
    if not path.exists():
        return None
    return HinglishABSA.load(str(path))


model = load_model()
st.title("🛒 Hinglish Quick-Commerce Review Analyser")
st.caption("Aspect-based sentiment + sarcasm detection for Romanised Hindi-English reviews "
           "(scikit-learn, CPU only).")

if model is None:
    st.error("No trained model found. Run `notebooks/hinglish_absa.ipynb` first (it saves models/absa_model.joblib).")
    st.stop()

mode = model.meta.get("data_mode", "unknown")
if mode != "REAL":
    st.warning(f"⚠️ Model trained in **{mode}** mode (synthetic smoke-test data, "
               f"{model.meta.get('n_reviews', '?')} reviews). Predictions are illustrative only. "
               "Train on your labelled data to get a real model.")
else:
    st.info(f"Aspect-sentiment model: {model.meta.get('n_reviews')} labelled Play Store reviews / "
            f"{model.meta.get('n_instances')} aspect instances ({model.meta.get('best_sentiment_model')}). "
            f"Sarcasm model: those reviews + {model.meta.get('sarcasm_extra_train', 0)} extra examples "
            f"(public Swami et al. corpus + hand-written seed examples).")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Sarcasm threshold", 0.1, 0.9, float(model.sarcasm_threshold), 0.05)
    show_lime = st.checkbox("Also run LIME (slower, word-level)", value=False)
    st.markdown("---")
    st.markdown("**Try these**")
    examples = [
        "Delivery 8 min me aa gayi, bahut fast service 👍",
        "Wah bhai kya baat hai, doodh expired bhej diya 👏 great job",
        "delivery fast thi but doodh expired tha, packing bhi acchi thi",
        "Price thoda mehnga hai lekin delivery boy bahut polite tha aur app smooth chalta hai",
        "Superb, 15 min bola 90 min ke baad bhi 'rider assigned', kya system hai 😂",
    ]
    for e in examples:
        if st.button(e[:48] + ("…" if len(e) > 48 else ""), key=e):
            st.session_state["review"] = e

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Paste or type a Hinglish review", key="review", height=110,
                        placeholder="e.g. Delivery late thi lekin sabzi fresh thi, price thoda mehnga")
with col2:
    rating = st.selectbox("Star rating (optional)", ["none", 1, 2, 3, 4, 5], index=0)
    run = st.button("Analyse", type="primary", use_container_width=True)

if run and text.strip():
    model.sarcasm_threshold = threshold
    r = model.predict_review(text, rating=None if rating == "none" else float(rating))

    # ---- row 1: sarcasm + code-mixing
    c1, c2, c3 = st.columns(3)
    c1.metric("Sarcasm", "YES 🙃" if r.sarcastic else "no", f"p = {r.sarcasm_proba:.2f}")
    c2.metric("Code-Mixing Index", f"{r.cmi:.1f}", "0 = monolingual, 50 = fully mixed")
    c3.metric("Aspects found", len(r.aspects))

    # ---- row 2: per-aspect table
    st.subheader("Sentiment per aspect")
    rows = []
    for a in r.aspects:
        rows.append({"Aspect": ASPECT_LABEL.get(a.aspect, a.aspect),
                     "Sentiment": f"{EMOJI[a.sentiment_final]} {a.sentiment_final}"
                                  + ("  (flipped from positive because of sarcasm)" if a.corrected else ""),
                     "Confidence": f"{max(a.proba.values()):.2f}",
                     "Context window used": a.window})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- row 3: why?
    st.subheader("Why? — explanation")
    t1, t2, t3 = st.tabs(["Sentiment evidence", "Sarcasm evidence", "Normalisation & language tags"])
    with t1:
        for a in r.aspects:
            st.markdown(f"**{ASPECT_LABEL.get(a.aspect, a.aspect)}** → raw model output *{a.sentiment_raw}* "
                        f"(probabilities: " + ", ".join(f"{k} {v:.2f}" for k, v in a.proba.items()) + ")")
            contrib = ex.contributions(model.sentiment_model, a.window, n=8)
            contrib["feature"] = contrib["feature"].str.replace("text__char__", "char n-gram: ", regex=False) \
                .str.replace("text__word__", "word: ", regex=False).str.replace("lex__", "lexicon: ", regex=False)
            st.dataframe(contrib[["feature", "contribution"]].round(3), hide_index=True, use_container_width=True)
            if show_lime:
                with st.spinner("running LIME…"):
                    pred, lst = ex.lime_explain(model.sentiment_model, a.window, num_samples=300)
                st.write("LIME word weights towards", pred, ":", {w: round(v, 3) for w, v in lst})
    with t2:
        if model.sarcasm_model is None:
            st.write("Sarcasm model not trained.")
        else:
            inc = r.incongruity
            st.markdown("Incongruity features (what the sarcasm detector looks at):")
            st.dataframe(pd.DataFrame({"feature": list(inc.keys()), "value": [round(v, 2) for v in inc.values()]}),
                         hide_index=True, use_container_width=True, height=330)
            X = pd.DataFrame([{"text": text, "rating": None if rating == "none" else float(rating)}])
            c = ex.contributions(model.sarcasm_model, X, target_class=1, n=10)
            c["feature"] = c["feature"].str.replace("incong__", "incongruity: ", regex=False) \
                .str.replace("text__tfidf__char__", "char n-gram: ", regex=False) \
                .str.replace("text__tfidf__word__", "word: ", regex=False)
            st.markdown("Top contributions **towards sarcastic**:")
            st.dataframe(c[["feature", "contribution"]].round(3), hide_index=True, use_container_width=True)
    with t3:
        st.code(r.normalized, language=None)
        st.write(" ".join(f"{t}/{l}" for t, l in zip(r.tokens, r.langs)))

    with st.expander("Raw JSON"):
        st.json(json.loads(json.dumps(r.to_dict(), default=float)))
elif run:
    st.warning("Type a review first.")
