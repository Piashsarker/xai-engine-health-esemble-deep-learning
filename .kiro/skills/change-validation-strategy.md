# Skill: Change Validation Strategy

## Description
Modify the cross-validation or train-test split strategy in VEHMS.

## Usage
Ask: "Change CV to 10-fold" or "Use stratified k-fold" or "Change test split to 30%"

## Train-Test Split Options

### Change Split Ratio
```python
# In DataPreprocessor class or preprocessing cell
# Default is 80/20, change test_size for different ratios

# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED
)

# 90/10 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=RANDOM_SEED
)
```

### Always Use Stratified Split
```python
# Ensures class distribution is preserved
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # IMPORTANT: maintains class balance
    random_state=RANDOM_SEED
)
```

## Cross-Validation Options

### Standard K-Fold
```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
```

### Stratified K-Fold (Recommended for Classification)
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
```

### Change Number of Folds
```python
# 10-fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# 3-fold CV (faster, less robust)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
```

### Repeated K-Fold (More Robust Estimates)
```python
from sklearn.model_selection import RepeatedStratifiedKFold

# 5-fold repeated 3 times = 15 total fits
cv = RepeatedStratifiedKFold(
    n_splits=5, 
    n_repeats=3, 
    random_state=RANDOM_SEED
)
```

### Leave-One-Out (Small Datasets)
```python
from sklearn.model_selection import LeaveOneOut

cv = LeaveOneOut()  # Each sample is test set once
# Warning: Very slow for large datasets
```

### Time Series Split (Temporal Data)
```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
# Maintains temporal order, no shuffling
```

## Updating ModelEvaluator

### Change Default CV in cross_validate Method
```python
def cross_validate(self, model, X, y, cv: int = 5, scoring: list = None) -> dict:
    """
    Perform k-fold cross-validation.
    
    Args:
        cv: Number of folds (default: 5) or CV splitter object
    """
    if isinstance(cv, int):
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = cv
    
    # ... rest of method
```

### Using Custom CV in Evaluation
```python
# In evaluation cell
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

for name, model in models.items():
    cv_results = evaluator.cross_validate(
        model, X_train, y_train, 
        cv=cv,  # Pass custom CV splitter
        scoring=['accuracy', 'precision_weighted', 'f1_weighted']
    )
```

## Updating StackingClassifier CV

```python
# Change CV folds in stacking
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=10,  # Change from default 5 to 10
)
```

## Multiple Scoring Metrics

```python
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted',
    'roc_auc': 'roc_auc_ovr_weighted'
}

cv_results = cross_validate(
    model, X, y, 
    cv=cv, 
    scoring=scoring,
    return_train_score=True
)

# Access results
print(f"Test Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Test F1: {cv_results['test_f1'].mean():.4f}")
```

## Nested Cross-Validation (For Hyperparameter Tuning)

```python
from sklearn.model_selection import GridSearchCV

# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search with inner CV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=inner_cv,
    scoring='accuracy'
)

# Evaluate with outer CV
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy')
print(f"Nested CV Accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")
```

## Quick Reference

| Strategy | Code | Use Case |
|----------|------|----------|
| 5-Fold Stratified | `StratifiedKFold(n_splits=5)` | Default, balanced classes |
| 10-Fold Stratified | `StratifiedKFold(n_splits=10)` | More robust estimates |
| Repeated 5x3 | `RepeatedStratifiedKFold(5, 3)` | Most robust, slower |
| Leave-One-Out | `LeaveOneOut()` | Very small datasets |
| Time Series | `TimeSeriesSplit(n_splits=5)` | Temporal data |
