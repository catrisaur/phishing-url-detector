import streamlit as st
import pandas as pd
import joblib
import os
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Feature Extraction Function ===
def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path

    features = {}
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['count_dots'] = url.count('.')
    features['count_hyphens'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    features['count_equal'] = url.count('=')
    features['has_https'] = int(parsed.scheme == 'https')
    features['has_ip'] = int(bool(re.search(r'\\d+\\.\\d+\\.\\d+\\.\\d+', hostname)))
    features['count_subdomains'] = hostname.count('.') - 1
    shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly']
    features['is_shortened'] = int(any(service in url for service in shortening_services))
    return features

def extract_features_df(url):
    return pd.DataFrame([extract_features(url)])

# === Load or train model ===
model_path = "model/phishing_rf_model.pkl"

if not os.path.exists(model_path):
    st.info("Training model on first run...")
    df = pd.read_csv("dataset_phishing.csv")

    # Drop unneeded columns (based on previous logic)
    drop_cols = [
        'nb_or', 'ratio_nullHyperlinks', 'ratio_intRedirection', 'ratio_intErrors',
        'submit_email', 'sfh', 'url'
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    df["status"] = df["status"].apply(lambda x: '1' if x == 'phishing' else '0')
    X = df.drop(columns=["status"])
    y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(rf, model_path)
    model = rf
else:
    model = joblib.load(model_path)

# === Streamlit App UI ===
st.title("🔐 Phishing URL Detection App")
st.markdown("Enter a URL below to check if it's **phishing** or **legitimate**.")

# Input URL
url_input = st.text_input("Enter URL here:", placeholder="https://example.com/login")

if st.button("Analyze URL"):
    if url_input:
        try:
            input_df = extract_features_df(url_input)
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][int(prediction)]
            if prediction == '1' or prediction == 1:
                st.error(f"⚠️ The URL is **likely phishing**. Confidence: {proba:.2%}")
            else:
                st.success(f"✅ The URL is **likely legitimate**. Confidence: {proba:.2%}")
        except Exception as e:
            st.error(f"An error occurred while processing the URL: {e}")
    else:
        st.warning("Please enter a URL before submitting.")
