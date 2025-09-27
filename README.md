# Analysis and Application of the Caret Package for Classification and Regression

This project was developed as part of a comprehensive study on machine learning techniques using the R programming language and the caret package. The project demonstrates practical applications of classification and regression models on real-world datasets.

**Note: This project was developed in a team. My main contributions were working on the SVM classification model on the Alzheimer's disease dataset, which achieved training accuracy of 88.66% and testing accuracy of 84.38%.**

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Key Findings](#key-findings)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results](#results)

## Project Overview

This project focuses on exploring and comparing various machine learning algorithms for both classification and regression tasks using the caret package in R. The study includes:

1. **Data preprocessing and feature engineering**
2. **Implementation of multiple machine learning models**
3. **Comprehensive model evaluation and comparison**
4. **Performance optimization using cross-validation and hyperparameter tuning**

## Datasets

### 1. Alzheimer's Disease Dataset
- **Size**: 2,149 patients with 35 features
- **Purpose**: Classification task to predict Alzheimer's disease diagnosis
- **Features**: Demographics, lifestyle factors, medical history, clinical measurements, cognitive assessments, and symptoms
- **Target Variable**: Diagnosis (0: No Alzheimer's, 1: Alzheimer's)

### 2. Housing Prices Dataset
- **Size**: 545 observations with 13 variables
- **Purpose**: Regression task to predict house prices
- **Features**: Area, bedrooms, bathrooms, stories, and categorical features (mainroad, basement, etc.)
- **Target Variable**: Price (continuous)

## Project Structure

```
Projet R/
├── alzheimers_disease_data.csv
├── Housing.csv
├── Rapport Projet B.pdf
├── Models/
│   ├── Classification/
│   │   ├── Decision_Tree.R
│   │   ├── LogitBoost.R
│   │   ├── Random Forest.R
│   │   ├── SVM.Rmd
│   │   └── Xgboost.Rmd
│   └── Regression/
│       ├── GBM.R
│       ├── GLMNET.R
│       ├── LM.R
│       ├── RLM.R
│       └── SVM.R
└── README.md
```

## Models Implemented

### Classification Models
1. **Decision Trees** - Simple, interpretable tree-based classification
2. **Random Forest** - Ensemble method combining multiple decision trees
3. **XGBoost** - Gradient boosting framework for enhanced performance
4. **Support Vector Machine (SVM)** - Margin-based classification with RBF kernel
5. **LogitBoost** - Boosting algorithm based on logistic regression

### Regression Models
1. **Linear Regression (LM)** - Basic linear relationship modeling
2. **Robust Linear Regression (RLM)** - Linear regression resistant to outliers
3. **Support Vector Machine (SVM)** - Non-linear regression with kernel methods
4. **Elastic Net (GLMNET)** - Regularized regression combining L1 and L2 penalties
5. **Gradient Boosting Machines (GBM)** - Ensemble method for regression

## Key Findings

### Classification Results (Alzheimer's Dataset)
| Model | Preprocessing Method | Training Accuracy | Test Accuracy |
|-------|---------------------|-------------------|---------------|
| Decision Tree | PCA | 70.81% | 72.02% |
| SVM | PCA | 88.66% | 84.38% |
| Random Forest | - | 90.29% | 90.00% |
| XGBoost | Correlation | 94.08% | 94.25% |
| LogitBoost | Correlation | 95.27% | 94.41% |

### Regression Results (Housing Dataset)
| Model | RMSE | R² |
|-------|------|-----|
| Robust Linear Regression (RLM) | 5.25 × 10⁶ | 0.687 |
| Linear Regression (LM) | 5.25 × 10⁶ | 0.683 |
| SVM | 5.25 × 10⁶ | 0.638 |
| Elastic Net (GLMNET) | 1.15 × 10⁶ | 0.637 |
| Gradient Boosting Machines (GBM) | 1.17 × 10⁶ | 0.667 |

### Best Performing Models
- **Classification**: LogitBoost achieved the highest test accuracy of 94.41%
- **Regression**: Robust Linear Regression (RLM) achieved the best R² of 0.687

## Installation and Setup

### Prerequisites
- R (version 4.0 or higher)
- RStudio (recommended)

### Required R Packages
```r
install.packages(c(
  "caret",
  "e1071",
  "randomForest",
  "xgboost",
  "MASS",
  "glmnet",
  "gbm",
  "pROC",
  "corrplot",
  "ggplot2",
  "dplyr",
  "reshape2"
))
```

## Usage

### Running Classification Models
1. Load the Alzheimer's dataset
2. Navigate to `Models/Classification/`
3. Run the desired model script (e.g., `SVM.Rmd` for SVM classification)

### Running Regression Models
1. Load the Housing dataset
2. Navigate to `Models/Regression/`
3. Run the desired model script (e.g., `RLM.R` for robust linear regression)

### Data Preprocessing Steps
1. **Missing Value Analysis**: Check for and handle missing data
2. **Feature Selection**: Use correlation analysis or PCA for dimensionality reduction
3. **Data Normalization**: Center and scale features for optimal model performance
4. **Train-Test Split**: 80% training, 20% testing

### Model Evaluation
- **Classification**: Accuracy, ROC curves, DET curves, Confusion matrices
- **Regression**: RMSE, R², Residual analysis

## Results

### SVM Classification Performance (My Contribution)
The Support Vector Machine model with RBF kernel achieved:
- **Training Accuracy**: 88.66%
- **Test Accuracy**: 84.38%
- **Preprocessing**: Principal Component Analysis (PCA) retaining 85% of variance
- **Cross-validation**: 5-fold CV for hyperparameter tuning

The SVM model demonstrated good generalization with balanced performance between training and test sets, indicating minimal overfitting.

### Key Insights
1. **Feature Engineering Impact**: PCA preprocessing significantly improved SVM performance
2. **Model Robustness**: Ensemble methods (Random Forest, XGBoost) showed superior performance
3. **Hyperparameter Tuning**: Cross-validation was crucial for optimal model performance
4. **Data Quality**: Proper preprocessing and feature selection were essential for model success

## License

This project is developed for educational purposes as part of the Artificial Intelligence Engineering curriculum.

**Institution**: Euro-Mediterranean University of Fez (UEMF)

**Academic Year**: 2024-2025

**Date**: December 20, 2024

## References

1. Max Kuhn. The caret Package. Available at: https://topepo.github.io/caret/
2. Yasser H. Housing Prices Dataset. Available at: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
3. Rabie El Kharoua. Alzheimer's Disease Dataset. Available at: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset