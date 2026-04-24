# VEHMS Machine Learning Configuration Guide

## Overview

This steering document provides guidelines for configuring the Vehicle Engine Health Monitoring System (VEHMS) machine learning pipeline. Use this guide when modifying classifiers, ensemble configurations, or evaluation strategies.

---

## 1. Base Classifier Configuration

### Available Classifiers

| Classifier | Key Parameters | Best For |
|------------|----------------|----------|
| **LR (Logistic Regression)** | `max_iter`, `C`, `solver` | Binary/multiclass, baseline model |
| **KNN** | `n_neighbors`, `weights`, `metric` | Small datasets, non-linear boundaries |
| **LDA** | `solver`, `shrinkage` | Dimensionality reduction, linear boundaries |
| **GNB (Gaussian Naive Bayes)** | `var_smoothing` | Fast training, probabilistic output |
| **SVM** | `kernel`, `C`, `gamma` | High-dimensional data, clear margins |
| **DT (Decision Tree)** | `max_depth`, `min_samples_split` | Interpretable, feature importance |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | Feature importance, robust predictions |
| **AdaBoost** | `n_estimators`, `learning_rate` | Weak learner boosting |
| **GB (Gradient Boosting)** | `n_estimators`, `max_depth`, `learning_rate` | Sequential improvement |
| **XGBoost** | `n_estimators`, `max_depth`, `learning_rate` | Large datasets, high accuracy |
| **LightGBM** | `n_estimators`, `num_leaves`, `learning_rate` | Fast training, large datasets |
| **CatBoost** | `iterations`, `depth`, `learning_rate` | Categorical features |
| **Extra Trees** | `n_estimators`, `max_depth` | Reduced variance |

### Adding a New Classifier

```python
# In BaseClassifierModule.__init__, add to self.classifiers dict:
self.classifiers['NewClassifier'] = NewClassifierClass(
    param1=value1,
    param2=value2,
    random_state=RANDOM_SEED
)
```

### Hyperparameter Tuning Guidelines

- **KNN**: Start with k=5, try odd values 3-15
- **SVM**: Use RBF kernel first, tune C (0.1-100) and gamma ('scale', 'auto', or 0.001-1)
- **Random Forest**: n_estimators=100-500, max_depth=None or 10-50
- **XGBoost**: learning_rate=0.01-0.3, max_depth=3-10, n_estimators=100-1000

---

## 2. Stacked Ensemble Configuration

### Stack Levels

| Level | Description | Example |
|-------|-------------|---------|
| **Level 0** | Base classifiers | KNN, SVM, RF, AdaBoost, XGBoost |
| **Level 1** | First meta-learner | Logistic Regression on Level 0 predictions |
| **Level 2** | Second meta-learner (optional) | Another classifier on Level 1 output |

### Creating Custom Stacked Models

```python
# Single-level stacking (recommended)
stacked_model = StackingClassifier(
    estimators=[
        ('clf1', Classifier1()),
        ('clf2', Classifier2()),
        ('clf3', Classifier3()),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,  # Cross-validation folds for base predictions
    stack_method='auto',  # 'auto', 'predict_proba', 'decision_function', 'predict'
    passthrough=False  # Set True to include original features with meta-features
)

# Multi-level stacking
level1_stack = StackingClassifier(
    estimators=[('rf', RF()), ('xgb', XGB())],
    final_estimator=LogisticRegression(),
    cv=5
)

level2_stack = StackingClassifier(
    estimators=[
        ('level1', level1_stack),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

### Recommended Stacking Configurations

**Configuration A - Diversity Focus:**
- Base: KNN + SVM + RF + XGBoost (diverse algorithms)
- Meta: Logistic Regression

**Configuration B - Boosting Focus:**
- Base: AdaBoost + XGBoost + LightGBM + CatBoost
- Meta: Random Forest

### Existing Research Stacking Configurations

These configurations are based on existing research literature for comparison:

**ER-Configuration 1 - Tree & Instance Based:**
- Base: RF + SVM + GB + DT + KNN
- Meta: Logistic Regression

**ER-Configuration 2 - Linear & Boosting:**
- Base: LR + SVM + LDA + GB + AdaBoost
- Meta: Logistic Regression

**ER-Configuration 3 - Comprehensive (All 9 Classifiers):**
- Base: LR + KNN + SVM + LDA + GB + AdaBoost + DT + RF + GNB
- Meta: Logistic Regression

### Dynamic Classifier Selection

Use the `DynamicClassifierSelector` class to automatically select optimal classifier combinations:

```python
from dynamic_classifier_selector import DynamicClassifierSelector

# Initialize selector
selector = DynamicClassifierSelector(cv=5, random_state=42)

# Run all selection methods
results = selector.run_all_selection_methods(X_train, y_train, X_test, y_test, top_k=5)

# Available selection methods:
# 1. Performance-based: selector.select_by_performance(top_k=5)
# 2. Diversity-based: selector.select_by_diversity(y_test, top_k=5)
# 3. Combined: selector.select_combined(y_test, top_k=5, performance_weight=0.6)
# 4. Greedy forward: selector.greedy_forward_selection(X_train, y_train, X_test, y_test, max_classifiers=5)

# Create dynamic stack from selection
dynamic_stack = selector.create_dynamic_stack(results['combined'])
dynamic_stack.fit(X_train, y_train)
```

**Selection Method Comparison:**

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Performance-Based | Fast | Good | Quick selection |
| Diversity-Based | Fast | Good | Maximizing ensemble diversity |
| Combined | Fast | Better | Balanced approach (recommended) |
| Greedy Forward | Slow | Best | When accuracy is critical |

---

## 3. Evaluation and Validation Strategies

### Cross-Validation Options

| Method | Use Case | Code |
|--------|----------|------|
| **K-Fold** | Standard evaluation | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| **Stratified K-Fold** | Imbalanced classes | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| **Leave-One-Out** | Small datasets | `LeaveOneOut()` |
| **Repeated K-Fold** | Robust estimates | `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)` |
| **Time Series Split** | Temporal data | `TimeSeriesSplit(n_splits=5)` |

### Metrics Configuration

```python
# Multi-class classification metrics
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'roc_auc_ovr': 'roc_auc_ovr_weighted'
}

# Custom scorer example
from sklearn.metrics import make_scorer, cohen_kappa_score
kappa_scorer = make_scorer(cohen_kappa_score)
```

### Validation Best Practices

1. **Always use stratified splits** for classification tasks
2. **Set random_state** for reproducibility
3. **Use nested CV** for hyperparameter tuning + evaluation
4. **Report mean ± std** for cross-validation scores
5. **Hold out a final test set** that's never used during training/tuning

---

## 4. Feature Engineering Options

### Scaling Methods

| Method | When to Use |
|--------|-------------|
| `StandardScaler` | Most algorithms, normally distributed data |
| `MinMaxScaler` | Neural networks, bounded features needed |
| `RobustScaler` | Data with outliers |
| `PowerTransformer` | Skewed distributions |

### Feature Selection Methods

```python
# Variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)

# SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)

# Feature importance from tree models
importances = rf_model.feature_importances_
```

---

## 5. XAI Configuration

### SHAP Options

| Explainer | Use Case |
|-----------|----------|
| `TreeExplainer` | Tree-based models (RF, XGBoost, LightGBM) |
| `KernelExplainer` | Any model (slower, model-agnostic) |
| `LinearExplainer` | Linear models |
| `DeepExplainer` | Deep learning models |

### LIME Configuration

```python
lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=True,  # Set False for continuous explanations
    kernel_width=None,  # Auto-calculated, or set manually
    sample_around_instance=True
)
```

---

## 6. Quick Reference Commands

```python
# Change number of CV folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Add new classifier to ensemble
stacked_ensemble.estimators.append(('new_clf', NewClassifier()))

# Change meta-learner
stacked_model.final_estimator = RandomForestClassifier(n_estimators=100)

# Enable passthrough (include original features)
stacked_model.passthrough = True

# Change test split ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```
