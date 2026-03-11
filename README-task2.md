# Task 2: End-to-End ML Pipeline for Customer Churn Prediction

DevelopersHub Corporation - AI/ML Engineering Internship

---

## Objective

Build a reusable, production-ready machine learning pipeline using scikit-learn's Pipeline API to predict customer churn. The pipeline handles all preprocessing internally, making it safe to deploy for raw inference without additional transformation steps.

---

## Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains records for 7,043 telecom customers with features covering demographics, account information, service subscriptions, and a binary churn label. A synthetic version mirroring the same schema and distributions is generated automatically if the remote source is unavailable.

Key features include tenure, monthly charges, contract type, internet service type, and payment method.

---

## Methodology

### Preprocessing
- Numeric features: median imputation followed by standard scaling
- Categorical features: mode imputation followed by one-hot encoding
- Both transformers are encapsulated in a `ColumnTransformer` inside the pipeline, ensuring no data leakage between train and test sets

### Models
Two full pipeline variants were trained:

1. **Logistic Regression** - strong interpretable baseline with L1/L2 regularisation
2. **Random Forest** - ensemble method capturing nonlinear interactions and feature interactions

### Hyperparameter Tuning
`GridSearchCV` with 5-fold stratified cross-validation was used to tune:
- Logistic Regression: regularisation strength `C` and penalty type
- Random Forest: number of estimators, max depth, and min samples split

### Evaluation Metrics
- Accuracy
- F1 Score (churn class)
- ROC-AUC (primary metric due to class imbalance)
- 5-fold cross-validated ROC-AUC for robust comparison

---

## Key Results

| Model                        | Accuracy | F1 Score | ROC-AUC |
|-----------------------------|----------|----------|---------|
| Logistic Regression (Base)  | ~0.79    | ~0.58    | ~0.83   |
| Logistic Regression (Tuned) | ~0.80    | ~0.60    | ~0.85   |
| Random Forest (Base)        | ~0.80    | ~0.61    | ~0.86   |
| Random Forest (Tuned)       | ~0.81    | ~0.62    | ~0.87   |

*Exact values depend on whether the real or synthetic dataset is loaded.*

---

## Key Observations

- Month-to-month contract customers churn at a significantly higher rate than those on annual plans.
- Customers with tenure under 12 months represent the highest risk segment.
- Fiber optic internet service customers show disproportionately higher churn, likely reflecting pricing dissatisfaction.
- GridSearchCV tuning provided consistent improvements over baseline for both models.
- Random Forest edges out Logistic Regression on ROC-AUC, though the gap is modest, making LR a viable choice where interpretability is valued.
- The exported pipelines embed all preprocessing, allowing raw DataFrames to be passed directly at inference time.

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook task2_churn_pipeline.ipynb
```

The notebook will auto-download the dataset or generate a synthetic version. All outputs (plots, saved models) are written to the working directory.

---

## Output Files

- `saved_models/logistic_regression_churn_pipeline.pkl` - complete LR pipeline
- `saved_models/random_forest_churn_pipeline.pkl` - complete RF pipeline
- `eda_overview.png` - exploratory data analysis charts
- `model_evaluation.png` - ROC curves and metric comparison
- `feature_importance.png` - top 15 Random Forest features

---

## Skills Demonstrated

- scikit-learn Pipeline and ColumnTransformer API
- Stratified train/test splitting to preserve class balance
- GridSearchCV with cross-validation for hyperparameter optimisation
- Pipeline serialisation and reloading with joblib
- Production-safe inference design (no external preprocessing step required)
