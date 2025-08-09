import re
import json
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Phishing URL Detection", page_icon="🔐", layout="centered")

# ---- Config ----
DATASET_PATH = "dataset_phishing.csv"

# Columns your original dataset likely has and that we can derive from a raw URL.
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
    "nb_comma",
]

# Columns that were dropped in your notebook (ignore if not present)
DROP_COLS = [
    "nb_or", "ratio_nullHyperlinks", "ratio_intRedirection", "ratio_intErrors",
    "submit_email", "sfh", "url"
]

def extract_features_from_url(url: str) -> dict:
    """Build dataset-like lexical features from a raw URL string."""
    parsed = urlparse(url)
    host = parsed.netloc or ""
    full = url or ""

    def count(ch: str) -> int:
        return full.count(ch)

    # IP in hostname? (strict-ish)
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

@st.cache_resource(show_spinner=True)
def load_train_model():
    """Loads data, trains URL-lexical RandomForest once, and returns (model, feature_order)."""
    df = pd.read_csv(DATASET_PATH)

    # Match your notebook’s cleaning
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    if "status" not in df.columns:
        raise ValueError("dataset_phishing.csv must contain a 'status' column (phishing vs legitimate).")

    # Encode target
    df["status"] = df["status"].apply(lambda x: "1" if str(x).lower() == "phishing" else "0")

    # Only keep URL-lexical features that actually exist in this CSV
    feature_order = [c for c in URL_FEATURES_CANDIDATES if c in df.columns]
    if not feature_order:
        raise ValueError(
            "No expected URL-lexical features found in dataset. "
            "Ensure your CSV has columns like length_url, nb_dots, nb_hyphens, etc."
        )

    X = df[feature_order]
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, feature_order

def make_input_df(url: str, feature_order: list[str]) -> pd.DataFrame:
    row = extract_features_from_url(url)
    # Ensure exact order and missing columns filled with 0
    ordered = {col: row.get(col, 0) for col in feature_order}
    return pd.DataFrame([ordered])

# ---- UI ----
st.title("🔐 Phishing URL Detection")
st.write("This app trains a URL‑lexical Random Forest model on first load and classifies new URLs.")

with st.spinner("Loading/Training model (first run only)…"):
    model, feature_order = load_train_model()

url_input = st.text_input("Enter URL", placeholder="https://example.com/login")

if st.button("Analyze URL"):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        try:
            X = make_input_df(url_input.strip(), feature_order)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][int(pred)]
            if str(pred) == "1":
                st.error(f"⚠️ Likely phishing. Confidence: {proba:.2%}")
            else:
                st.success(f"✅ Likely legitimate. Confidence: {proba:.2%}")
            with st.expander("Show computed features"):
                st.dataframe(X.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error(f"Error while processing: {e}")
