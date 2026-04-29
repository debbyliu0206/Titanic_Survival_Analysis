"""
This script performs a comprehensive survival analysis of the Titanic dataset.
It includes data loading, exploratory data analysis (EDA), detailed preprocessing,
feature engineering, and a comparison between Logistic Regression, Random Forest,
and Gradient Boosting models. All plots are saved to the 'plots/' directory,
and model performance metrics are printed to stdout.
"""

# ... (rest of imports)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# --- Section 1: Imports & Constants ---
RANDOM_STATE = 42
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Section 2: Data Loading ---
print("--- Section 2: Data Loading ---")
df = sns.load_dataset('titanic')
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("-" * 30 + "\n")

# --- Section 3: EDA ---
print("--- Section 3: Exploratory Data Analysis ---")
survival_counts = df['survived'].value_counts()
survival_rate = df['survived'].mean()
print(f"Survival Counts:\n{survival_counts}")
print(f"Overall Survival Rate: {survival_rate:.2%}")

# Visualization 1: Survival rate by pclass and sex
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='pclass', y='survived', hue='sex')
plt.title('Survival Rate by Class and Sex')
plt.ylabel('Survival Rate')
plt.savefig(os.path.join(PLOTS_DIR, 'survival_by_class_and_sex.png'))
plt.close()
print(f"Visualization 1 saved to {PLOTS_DIR}/survival_by_class_and_sex.png")
print("-" * 30 + "\n")

# --- Section 4: Preprocessing ---
print("--- Section 4: Preprocessing ---")

# 1. Impute 'age' with grouped median by pclass and sex
df['age'] = df.groupby(['pclass', 'sex'], group_keys=False)['age'].apply(lambda x: x.fillna(x.median()))

# 2. Derive 'has_cabin' flag from 'deck', then drop 'deck'
# In seaborn dataset, 'deck' is the proxy for cabin information.
df['has_cabin'] = df['deck'].notnull().astype(int)

# 3. Impute 'embarked' with mode ('S')
df['embarked'] = df['embarked'].fillna('S')

# 4. Drop redundant, leaky, or free text columns
# 'deck' is redundant with has_cabin info, 'embark_town' with 'embarked', 
# 'alive' is the target, 'alone' is derived from sibsp/parch, 
# 'who'/'adult_male' are redundant with age/sex.
# 'class' is redundant with 'pclass'
cols_to_drop = ['deck', 'embark_town', 'alive', 'alone', 'who', 'adult_male', 'class']
# Check if columns exist before dropping to avoid errors
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Encode 'sex' -> {female: 1, male: 0}
df['sex'] = df['sex'].map({'female': 1, 'male': 0})

# Encode 'embarked' -> one-hot (C as reference)
df = pd.get_dummies(df, columns=['embarked'], drop_first=True, dtype=int)

# Final check for nulls
assert df.isnull().sum().sum() == 0, "There are still missing values in the dataset!"
print("Preprocessing complete. All missing values handled and features encoded.")

# Visualization 2: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Cleaned Features')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
plt.close()
print(f"Visualization 2 saved to {PLOTS_DIR}/correlation_heatmap.png")
print("-" * 30 + "\n")

# --- Section 5: Feature Matrix & Train/Test Split ---
# Features: pclass, sex, age, sibsp, parch, fare, has_cabin, embarked_Q, embarked_S
X = df.drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale for Logistic Regression only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Section 6: Model Training ---
print("--- Section 6: Model Training ---")

# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
gb.fit(X_train, y_train)

print("Models trained successfully.")
print("-" * 30 + "\n")

# --- Section 7: Evaluation ---
print("--- Section 7: Evaluation ---")

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Metrics table
metrics = {
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr), 
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_gb)
    ],
    'Weighted F1': [
        f1_score(y_test, y_pred_lr, average='weighted'), 
        f1_score(y_test, y_pred_rf, average='weighted'),
        f1_score(y_test, y_pred_gb, average='weighted')
    ]
}
metrics_df = pd.DataFrame(metrics)
print("Model Comparison Metrics:")
print(metrics_df.to_string(index=False, float_format='{:.4f}'.format))

print("\nClassification Report - Gradient Boosting:")
print(classification_report(y_test, y_pred_gb))

# Visualization 3: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=axes[0], cmap='Blues')
axes[0].set_title('Logistic Regression')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=axes[1], cmap='Greens')
axes[1].set_title('Random Forest')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_gb, ax=axes[2], cmap='Oranges')
axes[2].set_title('Gradient Boosting')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'))
plt.close()
print(f"Visualization 3 saved to {PLOTS_DIR}/confusion_matrices.png")
print("-" * 30 + "\n")

# --- Section 8: Feature Importance ---
print("--- Section 8: Feature Importance ---")
importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True)

# Visualization 4: Horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
plt.close()
print(f"Visualization 4 saved to {PLOTS_DIR}/feature_importance.png")
print("-" * 30 + "\n")

# --- Section 9: Conclusions ---
print("--- Section 9: Conclusions ---")
best_model_idx = metrics_df['Accuracy'].idxmax()
best_model = metrics_df.iloc[best_model_idx]['Model']
best_acc = metrics_df.iloc[best_model_idx]['Accuracy']

print(f"Summary:")
print(f"- The {best_model} model performed best with an accuracy of {best_acc:.4f}.")
print(f"- Confusion matrices indicate that models are generally effective at predicting survivors and non-survivors.")
print(f"- Top features identified by Random Forest: {feat_imp_df.iloc[-1]['Feature']}, {feat_imp_df.iloc[-2]['Feature']}.")
print("- Suggested next steps: Hyperparameter tuning, advanced feature engineering (e.g., family size), or trying XGBoost.")
