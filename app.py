"""
Streamlit app: Breast Cancer Diagnosis Classifier Comparison
--------------------------------------------------------------
Lets a user upload the provided test_data.csv (or any CSV with the same
30 feature columns + a 'diagnosis' label column), pick one of 5 trained
classifiers, and see accuracy/AUC/precision/recall/F1/MCC plus a
confusion matrix and classification report.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis Classifier",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_scaler():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_model(filename):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_precomputed_metrics():
    return pd.read_csv(MODEL_DIR / "metrics_comparison.csv")


scaler = load_scaler()
feature_names = load_feature_names()
precomputed_metrics = load_precomputed_metrics()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Controls")
st.sidebar.markdown(
    "Upload the provided **test_data.csv** (or a CSV with the same 30 "
    "feature columns plus a `diagnosis` column) and choose a model to "
    "evaluate."
)

uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

selected_model_name = st.sidebar.selectbox(
    "Select a model", list(MODEL_FILES.keys())
)

show_all = st.sidebar.checkbox("Compare ALL models on this test data", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic) — 30 numeric features, "
    "569 instances, binary target (1 = malignant, 0 = benign)."
)

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("Breast Cancer Diagnosis — Multi-Model Classifier")
st.markdown(
    "This app demonstrates **5 classification models** trained on the "
    "Breast Cancer Wisconsin (Diagnostic) dataset: Logistic Regression, "
    "Decision Tree, k-Nearest Neighbors, Naive Bayes, and a Random Forest "
    "ensemble. If no file is uploaded, the bundled test data is loaded by default."
)

preview_path = BASE_DIR / "test_data.csv"
if uploaded_file is None and preview_path.exists():
    st.info(
        "📄 No file was uploaded, so the bundled **test_data.csv** from this repo is being loaded automatically."
    )
    uploaded_file = preview_path

if uploaded_file is None:
    st.info(
        "👈 Upload **test_data.csv** from the GitHub repo using the sidebar "
        "to run the models."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Read + validate uploaded data
# ---------------------------------------------------------------------------
try:
    user_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded CSV: {e}")
    st.stop()

missing_cols = [c for c in feature_names if c not in user_df.columns]
if missing_cols:
    st.error(
        "Uploaded CSV is missing required feature columns:\n\n"
        + ", ".join(missing_cols)
    )
    st.stop()

has_labels = "diagnosis" in user_df.columns

X_raw = user_df[feature_names]
X_scaled = scaler.transform(X_raw)

st.subheader("📄 Data Preview")
if uploaded_file == preview_path:
    st.caption("Using the bundled repository dataset: test_data.csv")
else:
    st.caption("Using the uploaded CSV file")
st.dataframe(user_df.head(10), use_container_width=True)
st.caption(f"{user_df.shape[0]} rows × {user_df.shape[1]} columns")


# ---------------------------------------------------------------------------
# Helper: evaluate a single model
# ---------------------------------------------------------------------------
def evaluate_model(model_name, X_scaled, y_true=None):
    model = load_model(MODEL_FILES[model_name])
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    out = {"predictions": y_pred, "probabilities": y_proba}

    if y_true is not None:
        out["metrics"] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
    return out


y_true = user_df["diagnosis"] if has_labels else None

# ---------------------------------------------------------------------------
# Single-model view
# ---------------------------------------------------------------------------
if not show_all:
    st.subheader(f"🔎 Results — {selected_model_name}")
    result = evaluate_model(selected_model_name, X_scaled, y_true)

    pred_df = user_df.copy()
    pred_df["predicted_diagnosis"] = result["predictions"]
    pred_df["malignant_probability"] = result["probabilities"].round(4)

    if has_labels and "metrics" in result:
        m = result["metrics"]
        cols = st.columns(6)
        labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        for c, lbl in zip(cols, labels):
            c.metric(lbl, f"{m[lbl]:.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, result["predictions"])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign (0)", "Malignant (1)"],
                yticklabels=["Benign (0)", "Malignant (1)"], ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.markdown("**Classification Report**")
            report = classification_report(
                y_true, result["predictions"],
                target_names=["Benign (0)", "Malignant (1)"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3),
                         use_container_width=True)
    else:
        st.warning(
            "No `diagnosis` column found in the uploaded data — showing "
            "predictions only (no evaluation metrics available)."
        )

    st.markdown("**Predictions**")
    st.dataframe(pred_df, use_container_width=True)

# ---------------------------------------------------------------------------
# All-models comparison view
# ---------------------------------------------------------------------------
else:
    st.subheader("🔎 Comparison — All Models on Uploaded Data")

    if not has_labels:
        st.warning(
            "No `diagnosis` column found in the uploaded data — cannot "
            "compute metrics. Showing predictions from each model instead."
        )
        pred_table = user_df.copy()
        for name in MODEL_FILES:
            result = evaluate_model(name, X_scaled)
            pred_table[f"{name} (pred)"] = result["predictions"]
        st.dataframe(pred_table, use_container_width=True)
    else:
        rows = []
        cms = {}
        for name in MODEL_FILES:
            result = evaluate_model(name, X_scaled, y_true)
            row = {"ML Model Name": name, **result["metrics"]}
            rows.append(row)
            cms[name] = confusion_matrix(y_true, result["predictions"])

        comp_df = pd.DataFrame(rows).set_index("ML Model Name").round(4)
        st.dataframe(comp_df, use_container_width=True)

        best_model = comp_df["MCC"].idxmax()
        st.success(f"🏆 Best performing model on this data (by MCC): **{best_model}**")

        st.markdown("**Confusion Matrices**")
        n = len(cms)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
        if n == 1:
            axes = [axes]
        for ax, (name, cm) in zip(axes, cms.items()):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.subheader("📊 Reference: Metrics on the Original Held-Out Test Split")
st.caption(
    "These are the metrics computed once during training (see "
    "`model/train_models.py`) on an 80/20 stratified split of the full "
    "dataset — included here for reference alongside whatever you upload above."
)
st.dataframe(precomputed_metrics.set_index("ML Model Name"), use_container_width=True)
