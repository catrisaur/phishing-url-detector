# app.py
# ------------------------------------------------------------
# Phishing URL Detector (Notebook-style pipeline, simplified)
# - Mirrors the Data.ipynb steps (drop cols, encode target)
# - Uses URL-lexical features present in the dataset
# - Trains at runtime (no pickles), so works on Streamlit Cloud
# - Clean, beginner-friendly code + comments
# ------------------------------------------------------------

import re
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Phishing URL Detection", page_icon="🔐", layout="centered")
st.title("🔐 Phishing URL Detection (Notebook-Style Pipeline)")
st.write("This app replicates our **Data.ipynb** training steps, but trains at runtime to stay compatible with Streamlit Cloud.")

# -------------------------------
# 1) Config (matches your notebook)
# -------------------------------
DATASET_PATH = "dataset_phishing.csv"

# Columns your notebook dropped (ignore safely if missing)
DROP_COLS = [
    "nb_or", "ratio_nullHyperlinks", "ratio_intRedirection", "ratio_intErrors",
    "submit_email", "sfh", "url"
]

# URL-lexical features commonly present in your dataset
# (we will automatically take only those that actually exist in the CSV)
URL_FEATURES_CANDIDATES = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma"
]

# -------------------------------
# 2) Helper: feature extraction from raw URL
#    (names match the dataset's style)
# -------------------------------
def extract_features_from_url(url: str) -> dict:
    parsed = urlparse(url)
    host = parsed.netloc or ""
    full = url or ""

    # very small helper to count a character
    def count(ch: str) -> int:
        return full.count(ch)

    # simple IP detection for hostname
    is_ip = 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) else 0

    return {
        "length_url": len(full),
        "length_hostname": len(host),
        "ip": is_ip,
        "nb_dots": count("."),
        "nb_hyphens": count("-"),
        "nb_at": count("@"),
        "nb_qm": count("?"),
        "nb_and": count("&"),
        "nb_eq": count("="),
        "nb_underscore": count("_"),
        "nb_tilde": count("~"),
        "nb_percent": count("%"),
        "nb_slash": count("/"),
        "nb_star": count("*"),
        "nb_colon": count(":"),
        "nb_comma": count(","),
    }

def make_input_df(url: str, feature_order: list[str]) -> pd.DataFrame:
    row = extract_features_from_url(url)
    # Ensure exact order & fill any missing with 0
    ordered = {col: row.get(col, 0) for col in feature_order}
    return pd.DataFrame([ordered])

# -------------------------------
# 3) Train model (mirrors notebook)
#    - Drop columns
#    - Encode target
#    - Use URL-lexical features available
#    - RF model, simple split, show accuracy
# -------------------------------
@st.cache_resource(show_spinner=True)
def train_model_from_csv():
    # Load CSV (same as notebook)
    df = pd.read_csv(DATASET_PATH)

    # Drop notebook’s unused columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Check target column
    if "status" not in df.columns:
        raise ValueError("The dataset must have a 'status' column with values like 'phishing'/'legitimate'.")

    # Encode target like the notebook: phishing -> '1', else '0'
    df["status"] = df["status"].apply(lambda x: "1" if str(x).lower() == "phishing" else "0")

    # Pick URL-lexical features that actually exist in this dataset
    feature_order = [c for c in URL_FEATURES_CANDIDATES if c in df.columns]
    if not feature_order:
        raise ValueError(
            "No expected URL-lexical features found. "
            "Ensure your CSV has columns like length_url, nb_dots, nb_hyphens, etc."
        )

    X = df[feature_order]
    y = df["status"]

    # Simple split + RandomForest (same spirit as notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Quick accuracy to display (nice for demo/report)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Quick & simple feature importance
    importances = pd.Series(model.feature_importances_, index=feature_order).sort_values(ascending=False)

    return model, feature_order, acc, importances

with st.spinner("Training model (one-time per server)…"):
    model, feature_order, acc, importances = train_model_from_csv()

st.success(f"Model trained. Test accuracy: **{acc:.2%}**")
with st.expander("Show feature importances"):
    st.dataframe(importances.rename("importance"))

# -------------------------------
# 4) Inference UI (raw URL → features → predict)
# -------------------------------
url = st.text_input("Enter a URL", placeholder="https://example.com/login")

if st.button("Analyze URL"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        try:
            X_new = make_input_df(url.strip(), feature_order)
            pred = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0][int(pred)]

            if str(pred) == "1":
                st.error(f"⚠️ Likely **phishing**. Confidence: {proba:.2%}")
            else:
                st.success(f"✅ Likely **legitimate**. Confidence: {proba:.2%}")

            with st.expander("See computed features for this URL"):
                st.dataframe(X_new.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error(f"Error: {e}")
