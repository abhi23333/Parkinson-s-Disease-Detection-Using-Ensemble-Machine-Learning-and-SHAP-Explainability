# Parkinson’s Disease Detection using Ensemble Machine Learning

## Overview
This project predicts Parkinson’s disease using voice-based biomedical features from the UCI Parkinson’s dataset. It applies advanced ensemble machine learning techniques and explainable AI to achieve high classification performance.

## Features
- XGBoost, LightGBM, and CatBoost models
- Soft voting ensemble classifier
- SMOTE for class imbalance handling
- Feature scaling using StandardScaler
- SHAP-based model interpretability
- Stratified train-test split
- 10-fold cross-validation evaluation

## Dataset
- Source: UCI Machine Learning Repository
- Features: Voice frequency and variation measurements
- Target: Status (0 = Healthy, 1 = Parkinson’s)

## Workflow
1. Data Loading
2. Preprocessing (Scaling + SMOTE)
3. Model Training:
   - XGBoost
   - LightGBM
   - CatBoost
4. Ensemble Learning (Soft Voting)
5. Evaluation (Accuracy, Precision, Recall, F1-score)
6. Cross-Validation (10-Fold)
7. Explainability using SHAP

## Results
- Accuracy: ~94.7%
- Precision (Parkinson’s class): 0.97
- Recall (Parkinson’s class): 0.97
- F1-score: 0.97

## Key Strengths
- Strong performance on small biomedical dataset
- Ensemble learning improves stability
- Handles class imbalance using SMOTE
- Explainable AI using SHAP values

## Tech Stack
- Python
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Imbalanced-learn
- SHAP
- Google Colab

## Future Improvements
- Nested cross-validation for stronger validation
- Hyperparameter tuning using Optuna
- Feature selection using SHAP importance
- Deployment using Flask / Streamlit
