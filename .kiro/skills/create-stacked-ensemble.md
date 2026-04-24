# Skill: Create Custom Stacked Ensemble

## Description
Create a new stacked ensemble configuration with custom base classifiers and meta-learner.

## Usage
Ask: "Create a stacked ensemble with [classifiers]" or "Build a new stacking model using [config]"

## Steps

1. **Define base estimators** as a list of tuples:
```python
estimators = [
    ('name1', Classifier1(params)),
    ('name2', Classifier2(params)),
    ('name3', Classifier3(params)),
]
```

2. **Choose meta-learner**:
```python
# Options:
meta_learner = LogisticRegression(max_iter=1000, random_state=42)  # Default
meta_learner = RandomForestClassifier(n_estimators=50, random_state=42)
meta_learner = XGBClassifier(n_estimators=50, random_state=42)
meta_learner = SVC(probability=True, random_state=42)
```

3. **Create StackingClassifier**:
```python
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5,                    # CV folds for base predictions
    stack_method='auto',     # 'predict_proba', 'decision_function', 'predict'
    passthrough=False,       # True to include original features
    n_jobs=-1               # Parallel processing
)
```

4. **Train and evaluate**:
```python
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)
```

## Configuration Templates

### Diversity-Focused Stack
```python
# Combines different algorithm types for diversity
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
]
```

### Boosting-Focused Stack
```python
# All boosting algorithms
estimators = [
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]
```

### Tree-Focused Stack
```python
# All tree-based algorithms
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
]
```

### Minimal Stack (Fast)
```python
# Quick training, good performance
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, random_state=42)),
]
```

### Multi-Level Stack
```python
# Level 1
level1 = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# Level 2
level2 = StackingClassifier(
    estimators=[
        ('level1', level1),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
```

## Adding to StackedEnsemble Class

```python
def create_stacked_model_custom(self) -> StackingClassifier:
    """Create custom stacked model configuration"""
    estimators = [
        # Add your estimators here
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=self.cv
    )
```

## Best Practices
- Use diverse algorithms for better ensemble performance
- Keep CV folds consistent (typically 5)
- Always set random_state for reproducibility
- Consider passthrough=True if original features are informative
- Use n_jobs=-1 for faster training on multi-core systems
