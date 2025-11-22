import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

st.set_page_config(
    page_title="Advanced Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def prepare_nltk():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return True
    except:
        st.warning("NLTK data could not be loaded. Using fallback preprocessing.")
        return False

nltk_ok = prepare_nltk()

@st.cache_data
def read_csv(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["label", "text"])

        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        if "label" not in df or "text" not in df:
            lower = {c.lower(): c for c in df.columns}
            if "label" in lower and "text" in lower:
                df = df[[lower["label"], lower["text"]]]
                df.columns = ["label", "text"]
            else:
                return pd.DataFrame(columns=["label", "text"])

        df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
        return df[["label", "text"]]
    except:
        return pd.DataFrame(columns=["label", "text"])

def clean_text(t):
    if not isinstance(t, str):
        return ""

    t = t.lower()
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if not nltk_ok:
        return " ".join([w for w in t.split() if len(w) > 2])

    words = nltk.word_tokenize(t)
    stops = set(stopwords.words("english"))
    lem = WordNetLemmatizer()

    final = []
    for w in words:
        if w not in stops and len(w) > 2:
            final.append(lem.lemmatize(w))

    return " ".join(final)

def main():
    st.sidebar.info("Training with local file: Generated_Spam_Dataset.csv")
    default_path = r"D:\spam\Generated_Spam_Dataset.csv"
    file_path = st.sidebar.text_input("Dataset path:", value=default_path)

    df = read_csv(file_path)

    if df.empty:
        st.warning("Dataset is empty or missing required columns.")
        st.info("CSV must contain: label, text")
        return

    total = len(df)
    spam_total = int((df["label"] == "spam").sum())
    ham_total = total - spam_total

    st.sidebar.metric("Total", total)
    st.sidebar.metric("Spam", spam_total)
    st.sidebar.metric("Ham", ham_total)

    with st.spinner("Cleaning text..."):
        df["processed_text"] = df["text"].astype(str).apply(clean_text)

    counts = df["label"].value_counts().to_dict()
    labels_available = list(counts.keys())

    if len(labels_available) < 2:
        st.error(f"Not enough label categories. Found: {labels_available}")
        return

    stratify = None
    if all(v >= 2 for v in counts.values()):
        stratify = df["label"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df["processed_text"], df["label"], test_size=0.2,
            random_state=42, stratify=stratify
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            df["processed_text"], df["label"], test_size=0.2,
            random_state=42
        )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(C=10, max_iter=1000, class_weight="balanced"))
    ])

    with st.spinner("Training model..."):
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training error: {e}")
            return

    preds = model.predict(X_test)

    try:
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="spam", zero_division=0)
        rec = recall_score(y_test, preds, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, preds, pos_label="spam", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    except:
        acc = prec = rec = f1 = 0
        cm = np.array([[0, 0], [0, 0]])

    st.title("Advanced Spam Email Classifier")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall", f"{rec*100:.2f}%")
    c4.metric("F1 Score", f"{f1*100:.2f}%")

    if st.checkbox("Show Confusion Matrix"):
        st.dataframe(pd.DataFrame(
            cm,
            columns=["Predicted Ham", "Predicted Spam"],
            index=["Actual Ham", "Actual Spam"]
        ))

    if st.checkbox("Show Dataset Preview"):
        st.dataframe(df[["label", "text"]].head())

    st.divider()
    text_input = st.text_area("Enter a message:")

    if st.button("Classify"):
        if text_input.strip() == "":
            st.warning("Enter text first.")
            return

        processed = clean_text(text_input)

        try:
            pred = model.predict([processed])[0]
        except:
            st.error("Prediction failed.")
            return

        prob_spam = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([processed])[0]
                idx = list(model.classes_).index("spam")
                prob_spam = probs[idx]
            except:
                pass

        if pred == "spam":
            if prob_spam is not None:
                st.error(f"Spam detected ({prob_spam:.2%})")
            else:
                st.error("Spam detected")
        else:
            if prob_spam is not None:
                st.success(f"Ham ({1 - prob_spam:.2%})")
            else:
                st.success("Ham")

        top_feats = []
        try:
            names = model.named_steps["tfidf"].get_feature_names_out()
            tfidf_vec = model.named_steps["tfidf"].transform([processed])
            nz = tfidf_vec.nonzero()[1]
            vals = tfidf_vec.data
            coefs = model.named_steps["clf"].coef_.flatten()

            for i, pos in enumerate(nz):
                if pos < len(coefs):
                    top_feats.append((names[pos], vals[i] * coefs[pos]))

            top_feats.sort(key=lambda x: abs(x[1]), reverse=True)
            top_feats = top_feats[:5]
        except:
            top_feats = []

        if top_feats:
            df_feat = pd.DataFrame(top_feats, columns=["Feature", "Impact"])
            df_feat["Type"] = df_feat["Impact"].apply(lambda x: "Spam" if x > 0 else "Ham")
            df_feat["Impact"] = df_feat["Impact"].abs()
            st.dataframe(df_feat)


if __name__ == "__main__":
    main()
