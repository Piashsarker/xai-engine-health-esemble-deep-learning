# Vehicle Engine Health Monitoring System (VEHMS)

## Predictive Maintenance Framework using Stacked Ensemble Deep Learning

VEHMS is a machine learning framework that analyzes vehicle engine sensor data and environmental conditions to predict engine health status. The system classifies engine conditions into four categories: **Good**, **Minor**, **Moderate**, and **Critical**.

---

## Overview

This project implements a comprehensive predictive maintenance solution using stacked ensemble learning. By combining multiple base classifiers with meta-learning, the system achieves high accuracy in predicting engine health degradation before failures occur.

### Key Features

- **Multi-sensor data analysis**: Processes 6 engine sensor readings and 3 environmental factors
- **Stacked ensemble architecture**: Combines diverse classifiers for robust predictions
- **Dynamic classifier selection**: Automatically selects optimal classifier combinations
- **Explainable AI**: SHAP and LIME integration for model interpretability
- **Research comparison**: Benchmarks against existing research configurations

### Target Performance Metrics

| Metric | Target |
|--------|--------|
| AUC | ~0.9702 |
| Accuracy | ~0.9470 |
| Precision | ~0.9486 |
| RMSE | ~0.3355 |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VEHMS - Predictive Maintenance Framework                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Data Layer  │───▶│ Processing   │───▶│   Model      │                   │
│  │              │    │    Layer     │    │   Layer      │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Data Loader  │    │Data Cleaner  │    │Base Classifier│                  │
│  │Data Visualizer│   │Data Preproc  │    │Stacked Ensemble│                 │
│  └──────────────┘    │Feature Extr  │    │Model Evaluator│                  │
│                      └──────────────┘    └──────────────┘                   │
│                                                 │                           │
│                                                 ▼                           │
│                                          ┌──────────────┐                   │
│                                          │ XAI Layer    │                   │
│                                          │ SHAP + LIME  │                   │
│                                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stacked Ensemble Models

The framework implements multiple stacked ensemble configurations, each designed with different classifier combinations to optimize performance and diversity.

### Stacking Strategy

Stacking (stacked generalization) is a two-level ensemble technique:

1. **Level 0 (Base Classifiers)**: Multiple diverse classifiers trained on the original features
2. **Level 1 (Meta-Learner)**: A classifier that learns to combine base classifier predictions

The key insight is that different classifiers make different types of errors. By combining them intelligently, the ensemble can achieve better performance than any individual classifier.

```
                    ┌─────────────────────────────────────┐
                    │         Input Features (9)          │
                    │  Crankshaft, Overheating, Lubricant │
                    │  Misfires, Piston, Starter,         │
                    │  Temperature, Humidity, Altitude    │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │  Base     │   │  Base     │   │  Base     │
              │Classifier │   │Classifier │   │Classifier │
              │    1      │   │    2      │   │    N      │
              └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     Meta-Learner (Level 1)      │
                    │     Logistic Regression         │
                    │     (5-fold CV predictions)     │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │      Final Prediction           │
                    │  Good | Minor | Moderate | Critical │
                    └─────────────────────────────────┘
```

### Proposed Stacked Models

#### Stacked Model 1 (Primary)
**Configuration**: KNN + SVM + RF + AdaBoost + XGBoost → Logistic Regression

This is the primary model combining instance-based, kernel-based, and boosting methods for maximum diversity.

```
              ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
              │    KNN      │   │    SVM      │   │    RF       │
              │    k=5      │   │    RBF      │   │   n=100     │
              └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                     │                 │                 │
              ┌──────┴──────┐   ┌──────┴──────┐          │
              │  AdaBoost   │   │  XGBoost    │          │
              │   n=100     │   │   n=100     │          │
              └──────┬──────┘   └──────┬──────┘          │
                     │                 │                 │
                     └─────────────────┼─────────────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │   Meta-Learner: Logistic Reg      │
                     └───────────────────────────────────┘
```

#### Stacked Model 2
**Configuration**: RF + XGBoost + AdaBoost → Logistic Regression

A boosting-focused ensemble combining the three strongest tree-based methods.

#### Stacked Model 3
**Configuration**: KNN + SVM + RF → Logistic Regression

A minimal ensemble balancing simplicity with diversity across algorithm families.

### Existing Research Configurations (for Comparison)

| Model | Base Classifiers | Focus |
|-------|------------------|-------|
| ER-Stacked 1 | RF + SVM + GB + DT + KNN | Tree & Instance Based |
| ER-Stacked 2 | LR + SVM + LDA + GB + AdaBoost | Linear & Boosting |
| ER-Stacked 3 | All 9 Classifiers | Comprehensive |

---

## Dynamic Classifier Selection

The `DynamicClassifierSelector` component automatically selects optimal classifier combinations using research-based techniques:

### Selection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Performance-Based** | Selects top-k classifiers by accuracy | Quick selection |
| **Diversity-Based** | Maximizes ensemble diversity using Q-statistic | Reducing correlated errors |
| **Combined** | Balances performance (60%) and diversity (40%) | General use (recommended) |
| **Greedy Forward** | Iteratively adds classifiers that improve ensemble | Maximum accuracy |

### Diversity Measurement

The Q-statistic measures agreement between classifier pairs:
- **Q = 1**: Classifiers always agree
- **Q = 0**: Classifiers are independent
- **Q = -1**: Classifiers always disagree (maximum diversity)

Lower Q values indicate higher diversity, which is beneficial for ensemble performance.

### Usage Example

```python
from dynamic_classifier_selector import DynamicClassifierSelector

# Initialize selector
selector = DynamicClassifierSelector(cv=5, random_state=42)

# Run all selection methods
results = selector.run_all_selection_methods(X_train, y_train, X_test, y_test, top_k=5)

# Create dynamic stack from combined selection
dynamic_stack = selector.create_dynamic_stack(results['combined'])
dynamic_stack.fit(X_train, y_train)
print(f"Dynamic Stack Accuracy: {dynamic_stack.score(X_test, y_test):.4f}")
```

---

## Available Base Classifiers

| Classifier | Abbreviation | Key Parameters | Strengths |
|------------|--------------|----------------|-----------|
| Logistic Regression | LR | max_iter=1000 | Baseline, interpretable |
| K-Nearest Neighbors | KNN | n_neighbors=5 | Non-linear boundaries |
| Linear Discriminant Analysis | LDA | - | Dimensionality reduction |
| Gaussian Naive Bayes | GNB | - | Fast, probabilistic |
| Support Vector Machine | SVM | kernel='rbf' | High-dimensional data |
| Decision Tree | DT | - | Interpretable, feature importance |
| Random Forest | RF | n_estimators=100 | Robust, feature importance |
| AdaBoost | AdaBoost | n_estimators=100 | Weak learner boosting |
| Gradient Boosting | GB | n_estimators=100 | Sequential improvement |
| XGBoost | XGBoost | n_estimators=100 | High accuracy, scalable |

---

## Dataset

### Input Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| Crankshaft | float | Crankshaft sensor reading | 4.0 - 21.0 |
| Overheating | float | Overheating indicator | 8.0 - 26.0 |
| Lubricant | float | Lubricant level/quality | 4.0 - 16.0 |
| Misfires | float | Engine misfire intensity | 2.5 - 11.0 |
| Piston | float | Piston sensor reading | 3.5 - 16.0 |
| Starter | float | Starter motor measurement | 250.0 - 450.0 |
| Temperature | float | Environmental temperature (°C) | 15.0 - 35.0 |
| Humidity | float | Environmental humidity (%) | 30.0 - 90.0 |
| Altitude | float | Environmental altitude (m) | 0.0 - 3000.0 |

### Target Variable

| Class | Description |
|-------|-------------|
| Good | Engine operating normally |
| Minor | Minor issues, routine maintenance recommended |
| Moderate | Moderate issues, maintenance required soon |
| Critical | Critical issues, immediate attention required |

---

## Project Structure

```
VEHMS/
├── README.md                              # This file
├── VEHMS_Predictive_Maintenance.ipynb     # Main notebook
├── dynamic_classifier_selector.py         # Dynamic selection component
├── dataset/
│   ├── augmented_data_with_environment.csv  # Primary dataset (10,000 samples)
│   └── engine_data.csv                      # Original engine data
└── .kiro/
    ├── specs/                             # Specification documents
    └── steering/                          # Configuration guides
```

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scipy
pip install matplotlib seaborn plotly
pip install scikit-learn xgboost
pip install shap lime
```

### Quick Start

1. Open `VEHMS_Predictive_Maintenance.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The notebook will:
   - Load and validate the dataset
   - Train base classifiers and stacked ensembles
   - Evaluate model performance
   - Generate XAI explanations

---

## Explainable AI (XAI)

The framework includes comprehensive explainability through:

### SHAP (SHapley Additive exPlanations)
- Global feature importance across all predictions
- Local explanations for individual predictions
- Dependence plots showing feature interactions

### LIME (Local Interpretable Model-agnostic Explanations)
- Instance-level explanations
- Feature contribution weights
- Human-readable decision rules

---

## Evaluation Metrics

The framework evaluates models using:

- **Accuracy**: Overall correct predictions
- **Precision**: Positive predictive value (weighted)
- **Recall**: True positive rate (weighted)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve (one-vs-rest)
- **RMSE**: Root mean square error
- **Confusion Matrix**: Detailed class-wise performance

---

## License

This project is for research and educational purposes.

---

## Acknowledgments

- Scikit-learn for machine learning implementations
- XGBoost for gradient boosting
- SHAP and LIME for explainability tools
