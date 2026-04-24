# Skill: Add Evaluation Metric

## Description
Add new evaluation metrics to the VEHMS model evaluation pipeline.

## Usage
Ask: "Add Cohen's Kappa metric" or "Include balanced accuracy" or "Add custom metric"

## Steps

1. **Import the metric** in the imports cell
2. **Add calculation method** to ModelEvaluator class
3. **Include in evaluate_model** method
4. **Update comparison table** to display new metric

## Common Metrics to Add

### Balanced Accuracy
```python
from sklearn.metrics import balanced_accuracy_score

def calculate_balanced_accuracy(self, y_true, y_pred) -> float:
    """Calculate balanced accuracy (useful for imbalanced classes)"""
    return balanced_accuracy_score(y_true, y_pred)
```

### Cohen's Kappa
```python
from sklearn.metrics import cohen_kappa_score

def calculate_kappa(self, y_true, y_pred) -> float:
    """Calculate Cohen's Kappa score"""
    return cohen_kappa_score(y_true, y_pred)
```

### Matthews Correlation Coefficient
```python
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(self, y_true, y_pred) -> float:
    """Calculate Matthews Correlation Coefficient"""
    return matthews_corrcoef(y_true, y_pred)
```

### Log Loss
```python
from sklearn.metrics import log_loss

def calculate_log_loss(self, y_true, y_proba) -> float:
    """Calculate logarithmic loss"""
    return log_loss(y_true, y_proba)
```

### Specificity (Per Class)
```python
def calculate_specificity(self, y_true, y_pred, class_idx) -> float:
    """Calculate specificity for a specific class"""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm.sum() - (cm[class_idx, :].sum() + cm[:, class_idx].sum() - cm[class_idx, class_idx])
    fp = cm[:, class_idx].sum() - cm[class_idx, class_idx]
    return tn / (tn + fp) if (tn + fp) > 0 else 0
```

### Top-K Accuracy
```python
from sklearn.metrics import top_k_accuracy_score

def calculate_top_k_accuracy(self, y_true, y_proba, k=2) -> float:
    """Calculate top-k accuracy"""
    return top_k_accuracy_score(y_true, y_proba, k=k)
```

### Hamming Loss
```python
from sklearn.metrics import hamming_loss

def calculate_hamming_loss(self, y_true, y_pred) -> float:
    """Calculate Hamming loss"""
    return hamming_loss(y_true, y_pred)
```

### Jaccard Score
```python
from sklearn.metrics import jaccard_score

def calculate_jaccard(self, y_true, y_pred) -> float:
    """Calculate Jaccard similarity score"""
    return jaccard_score(y_true, y_pred, average='weighted')
```

## Adding to ModelEvaluator Class

```python
class ModelEvaluator:
    def __init__(self):
        self.metrics_results = {}
    
    # ... existing methods ...
    
    # Add new metric method
    def calculate_new_metric(self, y_true, y_pred) -> float:
        """Calculate new metric"""
        return new_metric_function(y_true, y_pred)
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """Calculate all metrics for a model"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = {
            'RMSE': self.calculate_rmse(y_test, y_pred),
            'MAE': self.calculate_mae(y_test, y_pred),
            'Accuracy': self.calculate_accuracy(y_test, y_pred),
            'Precision': self.calculate_precision(y_test, y_pred),
            'AUC': self.calculate_auc(y_test, y_proba),
            # Add new metric here
            'NewMetric': self.calculate_new_metric(y_test, y_pred),
        }
        
        self.metrics_results[model_name] = metrics
        return metrics
```

## Custom Scorer for Cross-Validation

```python
from sklearn.metrics import make_scorer

# Create custom scorer
def custom_metric(y_true, y_pred):
    # Your custom calculation
    return score

custom_scorer = make_scorer(custom_metric, greater_is_better=True)

# Use in cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
```

## Multi-Metric Cross-Validation

```python
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted',
    'roc_auc': 'roc_auc_ovr_weighted',
    'balanced_accuracy': 'balanced_accuracy',
    'kappa': make_scorer(cohen_kappa_score),
}

cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric_name in scoring.keys():
    scores = cv_results[f'test_{metric_name}']
    print(f"{metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Updating Comparison Table

```python
def compare_models(self) -> pd.DataFrame:
    """Generate comparison table of all model metrics"""
    df = pd.DataFrame(self.metrics_results).T
    
    # Reorder columns to include new metric
    column_order = ['RMSE', 'MAE', 'Accuracy', 'Precision', 'AUC', 'NewMetric']
    df = df[column_order]
    
    # Sort by primary metric
    df = df.sort_values('Accuracy', ascending=False)
    
    return df
```

## Metric Reference Table

| Metric | Range | Best Value | Use Case |
|--------|-------|------------|----------|
| Accuracy | 0-1 | 1 | Balanced classes |
| Balanced Accuracy | 0-1 | 1 | Imbalanced classes |
| Precision | 0-1 | 1 | Minimize false positives |
| Recall | 0-1 | 1 | Minimize false negatives |
| F1 Score | 0-1 | 1 | Balance precision/recall |
| AUC-ROC | 0-1 | 1 | Ranking quality |
| Cohen's Kappa | -1 to 1 | 1 | Agreement beyond chance |
| MCC | -1 to 1 | 1 | Balanced measure |
| Log Loss | 0-∞ | 0 | Probability calibration |
| RMSE | 0-∞ | 0 | Regression-style error |
