# Bankruptcy Prediction using Financial Indicators

## 1. Introduction

### Problem Overview

Financial bankruptcy can severely impact investors, creditors, employees, and the overall economy. Accurately predicting bankruptcy ahead of time enables companies and stakeholders to take preventive measures and make informed decisions.

This project focuses on leveraging machine learning techniques to predict whether a company is likely to go bankrupt based on its historical financial indicators.

### Why Is Bankruptcy Prediction Important?

- **Risk Mitigation:** Helps financial institutions and investors assess and minimize risk.
- **Early Intervention:** Companies can restructure operations before insolvency occurs.
- **Improved Decision-Making:** Enables more accurate credit scoring and loan approvals.
- **Economic Stability:** Reduces systemic risk of widespread corporate failures.

### Dataset Description

- **Source:** [UCI Machine Learning Repository - Polish Companies Bankruptcy Data](http://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- **Data Span:** 5 years of historical financial information from Polish companies.
- **Features:** 64 numerical financial indicators (e.g., profitability ratios, debt ratios)
- **Target Variable:**
  - `0` → Non-bankrupt
  - `1` → Bankrupt

The dataset is structured to reflect real-world scenarios where a balanced, data-driven approach is necessary for robust financial forecasting.

---

## 2. Installation

To install all required Python libraries, run the following command:

```bash
pip install ucimlrepo xgboost seaborn matplotlib scikit-learn pandas numpy imblearn
```

---

## 3. Necessary Imports

```python
# Data Loading & EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Source
from ucimlrepo import fetch_ucirepo

# ML Models and Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel

# Misc
import warnings
warnings.filterwarnings('ignore')

# Plotting style
sns.set(style="whitegrid")
```

---

## 4. Data Loading

- Fetch dataset from UCI Machine Learning Repository
- Extract features and target variables

```python
polish_companies_bankruptcy = fetch_ucirepo(id=365)

X = polish_companies_bankruptcy.data.features
y = polish_companies_bankruptcy.data.targets

print("Metadata:", polish_companies_bankruptcy.metadata)
print("\nVariable Information:", polish_companies_bankruptcy.variables)

display(X.head())
display(y.head())

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

---

## 5. Exploratory Data Analysis (EDA)

- Analyze class distribution
- Visualize feature correlations
- Detect outliers using IQR
- Plot boxplots for key features

```python
print(y.value_counts())
y.value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Class Distribution (Bankrupt vs Non-Bankrupt)")
plt.xticks(ticks=[0, 1], labels=["Non-Bankrupt (0)", "Bankrupt (1)"], rotation=0)
plt.ylabel("Proportion")
plt.xlabel("Class")
plt.show()

corr = X.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=.5)
plt.title("Feature Correlation Heatmap")
plt.show()

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
outliers_count = outliers.sum().sort_values(ascending=False)
print(outliers_count)

X[X.columns[:10]].boxplot(rot=90)
plt.title("Boxplot of First 10 Features")
plt.show()
```

---

## 6. Data Preprocessing

- Handle missing values and drop unreliable columns
- Address class imbalance using SMOTE
- Feature scaling with StandardScaler

```python
# Drop column with many missing values
X = X.drop(columns='A37')
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())
```

---

## 7. Model Selection & Development

- Initial training with XGBoost
- Feature selection based on importance
- Hyperparameter tuning via Grid Search

```python
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train_res)

selector = SelectFromModel(xgb, threshold='median')
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

print(f"Reduced features from {X_train_scaled.shape[1]} to {X_train_sel.shape[1]}")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_xgb = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_xgb.fit(X_train_sel, y_train_res)

print("Best Parameters:", grid_xgb.best_params_)
print("Best F1 Score:", grid_xgb.best_score_)
```

---

## 8. Model Performance Metrics

- Evaluation using accuracy, precision, recall, F1-score, and ROC AUC

```python
best_model = grid_xgb.best_estimator_

y_pred = best_model.predict(X_test_sel)
y_proba = best_model.predict_proba(X_test_sel)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
```

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Non-Bankrupt (0) | 0.98      | 1.00   | 0.99     | 8263    |
| Bankrupt (1)     | 0.94      | 0.60   | 0.73     | 418     |
| **Accuracy**     |           |        | **0.98** | 8681    |
| **Macro Avg**    | 0.96      | 0.80   | 0.86     | 8681    |
| **Weighted Avg** | 0.98      | 0.98   | 0.98     | 8681    |

**ROC AUC Score:** `0.9734`

---

## 9. Interpretation of Results and Business Insights

The model effectively identifies most non-bankrupt companies while reasonably detecting bankrupt ones, which is crucial for minimizing financial risk. Although recall for bankrupt firms can be improved, the high precision reduces false alarms, helping businesses focus on genuine risk cases.

This predictive capability enables proactive decision-making such as targeted credit reviews and risk management, ultimately reducing potential losses.

---
