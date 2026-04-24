# VEHMS Experiment Workflow Guide

---
inclusion: auto
---

## Overview

This guide defines the workflow for running experiments with different configurations in the VEHMS predictive maintenance system.

---

## 1. Experiment Setup Checklist

Before running any experiment:

- [ ] Set `RANDOM_SEED = 42` (or your chosen seed) for reproducibility
- [ ] Verify dataset is loaded correctly (10,000 samples, 10 columns)
- [ ] Confirm train/test split ratio (default: 80/20)
- [ ] Document the experiment configuration
- [ ] Clear previous model artifacts if needed

---

## 2. Standard Experiment Template

```python
# ============================================================
# EXPERIMENT: [Name your experiment]
# Date: [Date]
# Objective: [What you're testing]
# ============================================================

# Configuration
EXPERIMENT_NAME = "experiment_001"
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Classifiers to test
CLASSIFIERS = ['KNN', 'SVM', 'RF', 'AdaBoost', 'XGBoost']

# Stacking configuration
STACK_CONFIG = {
    'base_estimators': ['RF', 'XGBoost', 'AdaBoost'],
    'meta_learner': 'LogisticRegression',
    'cv_folds': 5
}

# Metrics to track
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'rmse']
```

---

## 3. Comparing Different Configurations

### A. Classifier Comparison Experiment

```python
# Test different classifiers with same preprocessing
classifiers_to_compare = {
    'KNN_k3': KNeighborsClassifier(n_neighbors=3),
    'KNN_k5': KNeighborsClassifier(n_neighbors=5),
    'KNN_k7': KNeighborsClassifier(n_neighbors=7),
    'SVM_rbf': SVC(kernel='rbf', probability=True),
    'SVM_linear': SVC(kernel='linear', probability=True),
    'RF_100': RandomForestClassifier(n_estimators=100),
    'RF_200': RandomForestClassifier(n_estimators=200),
}

results = {}
for name, clf in classifiers_to_compare.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

results_df = pd.DataFrame(results).T.sort_values('accuracy', ascending=False)
```

### B. Stacking Configuration Experiment

```python
# Test different stacking configurations
stack_configs = {
    'Stack_Full': [('knn', KNN()), ('svm', SVM()), ('rf', RF()), ('ada', Ada()), ('xgb', XGB())],
    'Stack_Boosting': [('ada', Ada()), ('xgb', XGB()), ('gb', GradientBoosting())],
    'Stack_Trees': [('rf', RF()), ('et', ExtraTrees()), ('xgb', XGB())],
    'Stack_Minimal': [('rf', RF()), ('xgb', XGB())],
}

for config_name, estimators in stack_configs.items():
    stacked = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    # Train and evaluate...
```

### C. Cross-Validation Strategy Experiment

```python
# Compare different CV strategies
cv_strategies = {
    'KFold_5': KFold(n_splits=5, shuffle=True, random_state=42),
    'KFold_10': KFold(n_splits=10, shuffle=True, random_state=42),
    'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'RepeatedStratified': RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
}

for cv_name, cv in cv_strategies.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{cv_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 4. Results Documentation Template

```markdown
## Experiment Results: [Experiment Name]

### Configuration
- Dataset: augmented_data_with_environment.csv (10,000 samples)
- Features: 9 (Crankshaft, Overheating, Lubricant, Misfires, Piston, Starter, Temperature, Humidity, Altitude)
- Target: Decision (4 classes: Good, Minor, Moderate, Critical)
- Train/Test Split: 80/20
- Random Seed: 42

### Models Tested
1. [Model 1 name and config]
2. [Model 2 name and config]
...

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | AUC | RMSE |
|-------|----------|-----------|--------|-----|-----|------|
| Model 1 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Model 2 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

### Best Model
- Name: [Best model]
- Configuration: [Details]
- Performance: [Key metrics]

### Observations
- [Key finding 1]
- [Key finding 2]

### Next Steps
- [Recommendation 1]
- [Recommendation 2]
```

---

## 5. Hyperparameter Tuning Workflow

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search for thorough exploration
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Randomized Search for large parameter spaces
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
```

---

## 6. Model Persistence

```python
import joblib
import pickle
from datetime import datetime

# Save model with metadata
def save_experiment(model, metrics, config, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/{name}_{timestamp}.joblib"
    
    experiment_data = {
        'model': model,
        'metrics': metrics,
        'config': config,
        'timestamp': timestamp
    }
    
    joblib.dump(experiment_data, filename)
    print(f"Saved: {filename}")

# Load model
def load_experiment(filename):
    return joblib.load(filename)
```
