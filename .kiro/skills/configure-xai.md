# Skill: Configure Explainable AI (XAI)

## Description
Configure SHAP and LIME explainability settings for the VEHMS model.

## Usage
Ask: "Change SHAP explainer type" or "Configure LIME settings" or "Add new XAI visualization"

## SHAP Configuration

### Explainer Types

```python
import shap

# TreeExplainer - Fast, for tree-based models
shap_explainer = shap.TreeExplainer(model)

# KernelExplainer - Model-agnostic, slower
shap_explainer = shap.KernelExplainer(
    model.predict_proba, 
    shap.sample(X_train, 100)  # Background samples
)

# LinearExplainer - For linear models
shap_explainer = shap.LinearExplainer(model, X_train)

# DeepExplainer - For neural networks
shap_explainer = shap.DeepExplainer(model, X_train[:100])

# GradientExplainer - For neural networks (alternative)
shap_explainer = shap.GradientExplainer(model, X_train[:100])
```

### Choosing the Right Explainer

| Model Type | Recommended Explainer |
|------------|----------------------|
| Random Forest | TreeExplainer |
| XGBoost | TreeExplainer |
| LightGBM | TreeExplainer |
| SVM | KernelExplainer |
| Logistic Regression | LinearExplainer |
| Neural Network | DeepExplainer |
| Stacked Ensemble | KernelExplainer |
| Any model | KernelExplainer (slowest) |

### SHAP Calculation Settings

```python
# Limit samples for faster computation
max_samples = 200
if len(X_test) > max_samples:
    indices = np.random.choice(len(X_test), max_samples, replace=False)
    X_sample = X_test[indices]
else:
    X_sample = X_test

# Calculate SHAP values
shap_values = shap_explainer.shap_values(X_sample)
```

### SHAP Visualization Options

```python
# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)

# Bar plot (mean absolute values)
shap.summary_plot(shap_values, X_sample, plot_type='bar', feature_names=feature_names)

# Force plot (single prediction)
shap.force_plot(
    shap_explainer.expected_value[class_idx],
    shap_values[class_idx][sample_idx],
    X_sample[sample_idx],
    feature_names=feature_names,
    matplotlib=True
)

# Waterfall plot
shap.waterfall_plot(shap.Explanation(
    values=shap_values[class_idx][sample_idx],
    base_values=shap_explainer.expected_value[class_idx],
    data=X_sample[sample_idx],
    feature_names=feature_names
))

# Dependence plot
shap.dependence_plot(
    feature_idx,  # or feature name
    shap_values[class_idx],
    X_sample,
    feature_names=feature_names
)

# Decision plot
shap.decision_plot(
    shap_explainer.expected_value[class_idx],
    shap_values[class_idx],
    feature_names=feature_names
)
```

## LIME Configuration

### Basic Setup

```python
import lime
import lime.lime_tabular

lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)
```

### LIME Parameters

```python
lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    
    # Discretization settings
    discretize_continuous=True,  # False for continuous explanations
    discretizer='quartile',      # 'quartile', 'decile', 'entropy'
    
    # Sampling settings
    sample_around_instance=True,
    
    # Kernel settings
    kernel_width=None,  # Auto-calculated, or set manually (e.g., 0.75)
    
    # Other
    random_state=42,
    verbose=False
)
```

### LIME Explanation Generation

```python
# Generate explanation for single instance
explanation = lime_explainer.explain_instance(
    X_test[sample_idx],
    model.predict_proba,
    num_features=9,      # Number of features to show
    top_labels=4,        # Number of classes to explain
    num_samples=5000     # Samples for local approximation
)

# Get explanation for specific class
explanation.as_list(label=class_idx)

# Visualize
explanation.as_pyplot_figure(label=class_idx)

# Get prediction probabilities
explanation.predict_proba
```

### LIME Visualization Options

```python
# Matplotlib figure
fig = explanation.as_pyplot_figure(label=class_idx)
plt.title(f'LIME Explanation - {class_names[class_idx]}')
plt.tight_layout()
plt.show()

# HTML (for notebooks)
explanation.show_in_notebook(show_table=True, show_all=False)

# Save as HTML
explanation.save_to_file('lime_explanation.html')
```

## Updating XAIExplainer Class

### Change SHAP Explainer Type

```python
def initialize_shap(self, X_background: np.ndarray, explainer_type: str = 'kernel') -> None:
    """Initialize SHAP explainer with specified type"""
    
    if explainer_type == 'tree':
        self.shap_explainer = shap.TreeExplainer(self.model)
    elif explainer_type == 'kernel':
        # Sample background data
        if len(X_background) > 100:
            X_bg = shap.sample(X_background, 100)
        else:
            X_bg = X_background
        self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, X_bg)
    elif explainer_type == 'linear':
        self.shap_explainer = shap.LinearExplainer(self.model, X_background)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
```

### Add New SHAP Visualization

```python
def plot_shap_decision(self, sample_indices: list = None) -> None:
    """Generate SHAP decision plot"""
    if sample_indices is None:
        sample_indices = range(min(20, len(self.X_sample)))
    
    for class_idx, class_name in enumerate(self.class_names):
        plt.figure(figsize=(12, 8))
        shap.decision_plot(
            self.shap_explainer.expected_value[class_idx],
            self.shap_values[class_idx][sample_indices],
            feature_names=self.feature_names
        )
        plt.title(f'SHAP Decision Plot - {class_name}')
        plt.tight_layout()
        plt.show()
```

### Configure LIME Discretization

```python
def initialize_lime(self, class_names: list, discretize: bool = True, 
                    discretizer: str = 'quartile') -> None:
    """Initialize LIME explainer with custom settings"""
    self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=self.X_train,
        feature_names=self.feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=discretize,
        discretizer=discretizer,
        random_state=42
    )
```

## Performance Tips

1. **Reduce sample size** for faster SHAP computation:
```python
max_samples = 100  # Reduce from 200
```

2. **Use TreeExplainer** when possible (much faster than KernelExplainer)

3. **Limit LIME samples** for faster explanations:
```python
explanation = lime_explainer.explain_instance(
    X_instance,
    model.predict_proba,
    num_samples=1000  # Reduce from 5000
)
```

4. **Cache SHAP values** to avoid recalculation:
```python
# Save
np.save('shap_values.npy', shap_values)
# Load
shap_values = np.load('shap_values.npy', allow_pickle=True)
```
