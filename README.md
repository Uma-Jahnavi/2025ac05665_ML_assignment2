# Breast Cancer Diagnosis — Multi-Model Classification App

## a. Problem Statement

Breast cancer diagnosis is typically confirmed via a biopsy, where cell
nuclei from a fine needle aspirate (FNA) are measured and examined under a
microscope. The goal of this project is to build and compare multiple
machine learning classifiers that predict whether a breast mass is
**malignant (cancerous)** or **benign (non-cancerous)** from a set of
digitized measurements of cell nuclei, and to expose the trained models
through an interactive web app so predictions and performance can be
inspected on new/held-out data.

This is a **binary classification** problem:
`diagnosis = 1` → malignant, `diagnosis = 0` → benign.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set
**Source:** UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer`
(originally donated by Dr. William H. Wolberg, University of Wisconsin)

| Property | Value |
|---|---|
| Domain | Medical / Oncology |
| Task | Binary classification (malignant vs. benign) |
| Instances | 569 |
| Features | 30 numeric features (≥ 12 required) |
| Target | `diagnosis` (1 = malignant, 0 = benign) |
| Class balance | 212 malignant / 357 benign |
| Missing values | None |

The 30 features are computed from digitized images of a fine needle
aspirate (FNA) of a breast mass. For each of 10 real-valued base
measurements (radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, fractal dimension), the **mean**,
**standard error**, and **"worst"/largest** value were computed, giving
10 × 3 = 30 features in total.

Features were standardized (zero mean, unit variance) using
`StandardScaler` fit on the training split only, then applied to both
train and test splits to avoid data leakage.

An 80/20 stratified train/test split was used. The 20% held-out test
split (114 rows, with true labels) is saved as `test_data.csv` in this
repository and is what the Streamlit app expects you to upload.

## c. GitHub Repository Link

[`GitHub Link`](https://github.com/Uma-Jahnavi/2025ac05665_ML_assignment2.git)

## d. Models Used

Five classification models were trained on the same dataset, same
train/test split, and same scaled features:

1. Logistic Regression
2. Decision Tree Classifier (`max_depth=5`)
3. K-Nearest Neighbor Classifier (`k=7`)
4. Naive Bayes Classifier (Gaussian)
5. Ensemble Model — Random Forest (`n_estimators=300`, `max_depth=8`)

### Comparison Table (on the 20% held-out test split, 114 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9649 | 0.9944 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

*(Malignant = positive class. Numbers are reproducible by running
`model/train_models.py` with `random_state=42`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strongest all-round performer: highest accuracy (tied), highest AUC (0.996), and highest MCC. The classes are close to linearly separable once features are standardized, which suits a linear decision boundary well, and it generalizes better than the tree-based single model. |
| Decision Tree | Weakest model overall (lowest accuracy, AUC, F1, and MCC). A single shallow tree overfits to a few axis-aligned splits and misses the smoother boundary the other models capture; recall on the malignant class is noticeably lower, meaning it misses more actual cancer cases — the least desirable trait for a medical screening tool. |
| kNN | Solid performance, close to Logistic Regression and Random Forest. Because it relies on distance in feature space, it benefits directly from the `StandardScaler` step; without scaling it would perform noticeably worse due to features with different units/ranges. |
| Naive Bayes | Very high AUC (0.989) — it ranks/orders probabilities well — but lower accuracy, precision, and F1 than the top models. Its independence assumption between the 30 (often correlated) features hurts its hard-classification quality even though its probability ranking stays strong. |
| Random Forest (Ensemble) | Tied for the best accuracy and achieved perfect precision (1.0) on this test split — every case it predicted "malignant" was actually malignant — but its recall (0.9048) is slightly lower, meaning it was a bit more conservative and missed a couple of malignant cases. Averaging many trees clearly improves on the single Decision Tree across every metric. |
| **Overall Winner** | **Logistic Regression**, by MCC (0.9245) and AUC (0.9960), with **Random Forest (Ensemble)** essentially tied on accuracy/F1 and offering the best precision. For a screening context where missing a cancer case (false negative) is costly, Logistic Regression's higher recall (0.9286) gives it a slight practical edge over Random Forest's higher precision. |

## Live Streamlit App

[https://2025ac05665-ml.streamlit.app/](https://2025ac05665-ml.streamlit.app/)

The app supports:
- **Dataset upload (CSV):** upload `test_data.csv` (included in this repo) or any CSV with the same 30 feature columns, optionally including a `diagnosis` column for evaluation.
- **Model selection dropdown:** choose any one of the 5 trained models, or check "Compare ALL models" to see every model side by side.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, and MCC computed live on the uploaded data.
- **Confusion matrix & classification report:** shown per model (or per model in the comparison view).

---

## Repository Structure

```
project-folder/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── test_data.csv              # Held-out test data (114 rows, with labels)
└── model/
    ├── train_models.py            # Script: trains all 5 models & saves everything below
    ├── logistic_regression.pkl    # Trained Logistic Regression model
    ├── decision_tree.pkl          # Trained Decision Tree model
    ├── knn.pkl                    # Trained kNN model
    ├── naive_bayes.pkl            # Trained Gaussian Naive Bayes model
    ├── random_forest_ensemble.pkl # Trained Random Forest model
    ├── scaler.pkl                 # Fitted StandardScaler (applied to raw features before prediction)
    ├── feature_names.json         # Ordered list of the 30 expected feature columns
    └── metrics_comparison.csv     # The comparison table above, saved as CSV
```

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/Uma-Jahnavi/2025ac05665_ML_assignment2.git
cd project-folder

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain all models from scratch
python model/train_models.py

# 4. Run the app locally
streamlit run app.py
```