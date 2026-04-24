# Skill: Add New Classifier to VEHMS

## Description
Add a new base classifier to the VEHMS ensemble pipeline.

## Usage
Ask: "Add [ClassifierName] classifier to VEHMS" or "Include [ClassifierName] in the base classifiers"

## Steps

1. **Import the classifier** in the imports cell:
```python
from sklearn.xxx import NewClassifier
# or
from library import NewClassifier
```

2. **Add to BaseClassifierModule** in the `__init__` method:
```python
self.classifiers['NewName'] = NewClassifier(
    param1=value1,
    param2=value2,
    random_state=RANDOM_SEED  # Always include for reproducibility
)
```

3. **Update the training cell** to include the new classifier in evaluation.

4. **Verify** by running the base classifier training section.

## Common Classifiers to Add

### LightGBM
```python
from lightgbm import LGBMClassifier
self.classifiers['LightGBM'] = LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    random_state=RANDOM_SEED,
    verbose=-1
)
```

### CatBoost
```python
from catboost import CatBoostClassifier
self.classifiers['CatBoost'] = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=RANDOM_SEED,
    verbose=False
)
```

### Extra Trees
```python
from sklearn.ensemble import ExtraTreesClassifier
self.classifiers['ExtraTrees'] = ExtraTreesClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED
)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier
self.classifiers['GradientBoosting'] = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=RANDOM_SEED
)
```

### Neural Network (MLP)
```python
from sklearn.neural_network import MLPClassifier
self.classifiers['MLP'] = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=RANDOM_SEED
)
```

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
self.classifiers['NaiveBayes'] = GaussianNB()
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
self.classifiers['DecisionTree'] = DecisionTreeClassifier(
    max_depth=10,
    random_state=RANDOM_SEED
)
```

## Validation
After adding, verify:
- [ ] Classifier trains without errors
- [ ] Predictions are generated correctly
- [ ] Metrics are calculated properly
- [ ] Can be included in stacked ensemble
