# Titanic Survival Analysis

This project provides a self-contained, reproducible survival analysis of the Titanic dataset. It demonstrates data cleaning, exploratory data analysis (EDA), feature engineering, and a comparison of multiple machine learning models.

## Project Overview

The analysis uses the Titanic dataset (loaded directly via `seaborn`) to predict passenger survival. It compares three different modeling approaches:
1.  **Logistic Regression** (Baseline)
2.  **Random Forest**
3.  **Gradient Boosting**

## Key Features

-   **Automated Data Cleaning:** Handles missing values for 'Age' (grouped median) and 'Embarked' (mode).
-   **Feature Engineering:** Derives a `has_cabin` flag from deck information and performs one-hot encoding for categorical variables.
-   **Visualizations:** Generates and saves four analytical plots to the `plots/` directory.
-   **Model Comparison:** Provides accuracy and F1-score comparisons for all models.

## Results Summary

| Model | Accuracy | Weighted F1 |
| :--- | :--- | :--- |
| Logistic Regression | 0.8101 | 0.8083 |
| Random Forest | 0.7821 | 0.7804 |
| Gradient Boosting | 0.7877 | 0.7831 |

*Note: Results may vary slightly depending on environment, but Logistic Regression consistently performs well on this relatively small dataset.*

### Key Insights
-   **Top Features:** Survival was most strongly influenced by `sex` and `age`, as identified by the Random Forest feature importance analysis.
-   **Demographics:** Females and children had significantly higher survival rates across all passenger classes.

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
Run the main analysis script:
```bash
python titanic_analysis.py
```

All plots will be generated in the `plots/` folder:
-   `survival_by_class_and_sex.png`
-   `correlation_heatmap.png`
-   `confusion_matrices.png`
-   `feature_importance.png`

## Dependencies
-   numpy
-   pandas
-   matplotlib
-   seaborn
-   scikit-learn
