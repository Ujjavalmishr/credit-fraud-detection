# app/streamlit.py
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# Path fix for importing scripts
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.main_baseline import (
    time_split,
    train_baseline,
    evaluate,
    plot_class_distribution,
    plot_amount_distribution,
)

from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="💳 Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud — Interactive Dashboard")

# -----------------------------
# File uploader
# -----------------------------
data_file = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
path = data_file if data_file else "./data/creditcard.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data(path)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Baseline Model", "Compare Saved Models"])

# =====================================================
# 📊 Exploratory Data Analysis (EDA)
# =====================================================
if page == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10))

    st.subheader("1. Class Imbalance")
    st.pyplot(plot_class_distribution(df))

    st.subheader("2. Transaction Amount Distribution (Fraud vs Non-Fraud)")
    st.pyplot(plot_amount_distribution(df))

    st.subheader("3. PCA Projection (Fraud vs Non-Fraud)")
    # -----------------------------
    # PCA Section
    # -----------------------------
    features = df.drop(columns=["Class", "Time"]) if "Time" in df.columns else df.drop(columns=["Class"])
    labels = df["Class"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(features_scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Class"] = labels.values

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Class",
        palette={0: "blue", 1: "red"},
        alpha=0.5,
        ax=ax
    )
    ax.set_title("PCA Projection of Transactions")
    st.pyplot(fig)

# =====================================================
# 🤖 Baseline Model
# =====================================================
elif page == "Baseline Model":
    st.header("Baseline Model Results")

    # -----------------------------
    # Train & evaluate
    # -----------------------------
    FEATURES = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    train, test = time_split(df)
    model = train_baseline(train, FEATURES)
    results = evaluate(model, test, FEATURES)

    y_true, y_proba = results["y_true"], results["y_proba"]

    # -----------------------------
    # PR curve
    # -----------------------------
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    st.subheader("Precision–Recall Curve")
    st.line_chart(pd.DataFrame({"precision": prec[:-1], "recall": rec[:-1]}))

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # Threshold slider + confusion matrix
    # -----------------------------
    st.sidebar.subheader("⚙️ Threshold Control")
    threshold = st.sidebar.slider(
        "Decision threshold", 0.0, 1.0, float(results["threshold"]), 0.01
    )

    pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    st.subheader("Metrics at current threshold")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud rate (test)", f"{(y_true.mean()*100):.3f}%")
    col2.metric("TP / FN", f"{tp} / {fn}")
    col3.metric("FP / TN", f"{fp} / {tn}")

    # Confusion matrix heatmap
    st.subheader("Confusion Matrix Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------
    # Top suspicious transactions
    # -----------------------------
    st.subheader("Top suspicious transactions")
    idx = np.argsort(-y_proba)[:50]
    top_suspicious = test.iloc[idx].assign(score=y_proba[idx])
    st.dataframe(top_suspicious)

# =====================================================
# 🏆 Compare Saved Models
# =====================================================
elif page == "Compare Saved Models":
    st.header("Compare Trained Models")

    # Look for saved models
    model_dir = "./models"
    available_models = [
        f for f in os.listdir(model_dir) if f.endswith(".pkl")
    ] if os.path.exists(model_dir) else []

    if not available_models:
        st.warning("No trained models found in `models/`. Run training first.")
    else:
        model_choice = st.selectbox("Choose a model", available_models)

        # Load model
        model = joblib.load(os.path.join(model_dir, model_choice))

        # Split train/test
        FEATURES = [c for c in df.columns if c.startswith("V")] + ["Amount"]
        train, test = time_split(df)
        X_test, y_true = test[FEATURES], test["Class"]

        # Predict
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        st.subheader(f"ROC Curve — {model_choice}")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Confusion Matrix")
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"], ax=ax)
        st.pyplot(fig)
