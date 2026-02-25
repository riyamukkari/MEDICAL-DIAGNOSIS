# Medical Diagnosis ML Pipeline  
Breast Cancer Classification

## Introduction

This project implements a complete machine learning pipeline for breast cancer classification using the Breast Cancer Wisconsin dataset from scikit-learn. The objective is to classify tumors as malignant or benign using multiple supervised learning models within a structured and reproducible workflow.

## Objective

The primary goal is to build and compare multiple classification models to determine the most effective approach for accurate and reliable breast cancer diagnosis. Emphasis is placed on performance evaluation using cross-validation and ROC-AUC metrics.

## Dataset

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Total Samples: 569
- Features: 30 numerical features derived from digitized tumor images
- Target Variable:
  - 0 → Malignant
  - 1 → Benign

## Data Preprocessing

- Converted dataset into a pandas DataFrame for structured analysis
- Performed stratified train-test split (80/20) to maintain class distribution
- Applied feature scaling using StandardScaler
- Used PCA (Principal Component Analysis) with 10 components for dimensionality reduction
- Integrated preprocessing steps into sklearn Pipelines to prevent data leakage

## Exploratory Data Analysis

- Examined dataset shape and feature structure
- Verified class balance before model training
- Evaluated classification performance using precision, recall, F1-score, and ROC-AUC
- Compared cross-validation accuracy and AUC across models

## Model Building

Three models were implemented using pipelines:

1. Logistic Regression  
2. Random Forest Classifier  
3. Support Vector Machine  

### Validation Strategy

- Stratified 5-Fold Cross-Validation
- Evaluation Metrics:
  - Accuracy
  - ROC-AUC Score
  - Classification Report

### Results Summary

- Logistic Regression and SVM achieved the highest Test AUC (~0.995)
- Random Forest performed slightly lower but maintained strong predictive capability
- All models demonstrated high classification performance

## Conclusion

The pipeline-based approach ensures clean preprocessing, reproducibility, and fair model comparison. Logistic Regression and Support Vector Machine achieved the best overall performance based on ROC-AUC. This project demonstrates the effectiveness of structured ML pipelines in medical diagnosis classification tasks.
