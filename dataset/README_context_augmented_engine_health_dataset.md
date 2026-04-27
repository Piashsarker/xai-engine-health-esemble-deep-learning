# Context-Augmented Engine Health Dataset

## Dataset Overview

**File**: `context_augmented_engine_health_dataset.csv`  
**Samples**: 12,200 rows  
**Features**: 26 columns (24 features + 1 time index + 1 target)  
**Target**: Decision (4-class classification)

This dataset extends the original VEHMS engine health data with rich contextual information including environmental scenarios, fault progression stages, and sensor noise levels.

---

## Target Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Critical | 3,063 | 25.1% |
| Good | 3,025 | 24.8% |
| Minor | 3,062 | 25.1% |
| Moderate | 3,050 | 25.0% |

**Balanced dataset** - approximately equal distribution across all 4 health states.

---

## Feature Categories

### 1. Time Index
| Feature | Description |
|---------|-------------|
| `Time` | Timestamp identifier (t1, t2, ...) |

### 2. Engine Sensor Features (6)
| Feature | Description | Range |
|---------|-------------|-------|
| `RPM` | Engine revolutions per minute | ~2400-3500 |
| `Engine_Temp` | Engine temperature (°C) | ~90-125 |
| `Oil_Pressure` | Oil pressure reading | ~2.2-3.6 |
| `Vibration` | Vibration intensity | ~0.3-0.7 |
| `Fuel_Rate` | Fuel consumption rate | ~6-9 |

### 3. Original VEHMS Features (6)
| Feature | Description | Range |
|---------|-------------|-------|
| `Crankshaft` | Crankshaft sensor reading | ~8-14 |
| `Overheating` | Overheating indicator | ~14-20 |
| `Lubricant` | Lubricant level/quality | ~8-12 |
| `Misfires` | Engine misfire intensity | ~3-7 |
| `Piston` | Piston sensor reading | ~8-13 |
| `Starter` | Starter motor measurement | ~325-380 |

### 4. Environmental Features (3)
| Feature | Description | Range |
|---------|-------------|-------|
| `Temperature` | Ambient temperature (°C) | ~19-30 |
| `Humidity` | Ambient humidity (%) | ~50-85 |
| `Altitude` | Operating altitude (m) | ~450-2700 |

### 5. Derived/Engineered Features (4)
| Feature | Description |
|---------|-------------|
| `Load_Percent` | Engine load percentage |
| `Temp_Gradient` | Temperature change rate |
| `Vibration_Rolling` | Rolling average of vibration |
| `Env_Stress_Index` | Combined environmental stress score |

### 6. Context Features (6)
| Feature | Description | Values |
|---------|-------------|--------|
| `Fault_Simulated` | Whether fault was simulated | 0, 1 |
| `Sensor_Noise_Level` | Noise level in sensors | Low, Medium, High |
| `Env_Scenario` | Environmental scenario | Normal, Desert, Arctic, Tropical, Mountain, Coastal |
| `Fault_Progression_Stage` | Stage of fault progression | None, Normal, Early, Mid, Severe |
| `Progression_Sequence_ID` | Sequence identifier | -1 (none), 1+ |
| `Has_Missing_Values` | Missing value indicator | 0, 1 |
| `Missing_Value_Count` | Count of missing values | 0+ |

---

## Environmental Scenario Distribution

| Scenario | Count | Description |
|----------|-------|-------------|
| Normal | 11,092 | Standard operating conditions |
| Arctic | 224 | Cold weather conditions |
| Coastal | 224 | High humidity, salt air |
| Desert | 224 | High temperature, low humidity |
| Mountain | 220 | High altitude, low pressure |
| Tropical | 216 | High temperature and humidity |

---

## Fault Progression Distribution

| Stage | Count | Description |
|-------|-------|-------------|
| None | 11,000 | No fault progression |
| Normal | 300 | Normal operation baseline |
| Early | 300 | Early fault indicators |
| Mid | 300 | Mid-stage fault development |
| Severe | 300 | Severe fault condition |

---

## Quick Start: Existing Research Stack Models

```python
from vehms import DataLoader, DataPreprocessor, RANDOM_SEED
from vehms.existing_research_stacked_ensemble import ExistingResearchStackedEnsemble
from vehms.model_evaluator import ModelEvaluator

# Load and preprocess data
loader = DataLoader()
df = loader.load_dataset('dataset/context_augmented_engine_health_dataset.csv')

preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

# Train Existing Research Stacked Models
er_stacked = ExistingResearchStackedEnsemble(cv=5, random_state=RANDOM_SEED)

# ER-Stacked Model 1: RF + SVM + GB + DT + KNN
er_stacked.train_stacked_model('ER-Stacked Model 1', 
                                er_stacked.create_er_stacked_model_1(), 
                                X_train, y_train)

# ER-Stacked Model 2: LR + SVM + LDA + GB + AdaBoost
er_stacked.train_stacked_model('ER-Stacked Model 2', 
                                er_stacked.create_er_stacked_model_2(), 
                                X_train, y_train)

# ER-Stacked Model 3: All 9 classifiers
er_stacked.train_stacked_model('ER-Stacked Model 3', 
                                er_stacked.create_er_stacked_model_3(), 
                                X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
for name, model in er_stacked.get_all_models().items():
    evaluator.evaluate_model(model, X_test, y_test, name)

evaluator.compare_models()
```

---

## Quick Start: Dynamic Stack Models

```python
from vehms import DataLoader, DataPreprocessor, RANDOM_SEED
from vehms.dynamic_classifier_selector import DynamicClassifierSelector
from vehms.model_evaluator import ModelEvaluator

# Load and preprocess data
loader = DataLoader()
df = loader.load_dataset('dataset/context_augmented_engine_health_dataset.csv')

preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

# Dynamic Classifier Selection
selector = DynamicClassifierSelector(cv=5, random_state=RANDOM_SEED)

# Run all selection methods (Performance, Diversity, Combined)
results = selector.run_all_selection_methods(X_train, y_train, X_test, y_test, top_k=5)

# Create and train dynamic stacks
evaluator = ModelEvaluator()

for method, selected_classifiers in results.items():
    stack = selector.create_dynamic_stack(selected_classifiers)
    stack.fit(X_train, y_train)
    evaluator.evaluate_model(stack, X_test, y_test, f'DS-Stack {method.title()}')
    print(f"DS-Stack {method.title()}: {selected_classifiers}")

evaluator.compare_models()
```

---

## Model Configurations

### Existing Research Stacked Models

| Model | Base Classifiers | Meta-Learner |
|-------|------------------|--------------|
| **ER-Stacked Model 1** | RF, SVM, GB, DT, KNN | Logistic Regression |
| **ER-Stacked Model 2** | LR, SVM, LDA, GB, AdaBoost | Logistic Regression |
| **ER-Stacked Model 3** | LR, KNN, SVM, LDA, GB, AdaBoost, DT, RF, GNB | Logistic Regression |

### Dynamic Stack Selection Methods

| Method | Strategy | Best For |
|--------|----------|----------|
| **Performance** | Top-k by individual accuracy | Quick selection |
| **Diversity** | Minimize Q-statistic (maximize diversity) | Ensemble diversity |
| **Combined** | 60% performance + 40% diversity | Balanced approach (recommended) |
| **Greedy Forward** | Iteratively add best improving classifier | Maximum accuracy |

---

## Feature Selection for Models

When using this dataset, you may want to select specific feature subsets:

```python
# Original VEHMS features only (for comparison with original research)
vehms_features = ['Crankshaft', 'Overheating', 'Lubricant', 'Misfires', 
                  'Piston', 'Starter']

# VEHMS + Environmental features
vehms_env_features = vehms_features + ['Temperature', 'Humidity', 'Altitude']

# All numeric features (excluding context/categorical)
all_numeric = ['RPM', 'Engine_Temp', 'Oil_Pressure', 'Vibration', 'Fuel_Rate',
               'Crankshaft', 'Overheating', 'Lubricant', 'Misfires', 'Piston', 
               'Starter', 'Temperature', 'Humidity', 'Altitude', 'Load_Percent',
               'Temp_Gradient', 'Vibration_Rolling', 'Env_Stress_Index']

# Select features
X = df[all_numeric]
y = df['Decision']
```

---

## Expected Performance Benchmarks

Based on similar datasets:

| Model Type | Expected Accuracy | Expected AUC | Expected RMSE |
|------------|-------------------|--------------|---------------|
| ER-Stacked Model 1 | 95-97% | 0.98-0.99 | 0.35-0.40 |
| ER-Stacked Model 2 | 95-97% | 0.98-0.99 | 0.35-0.40 |
| ER-Stacked Model 3 | 96-98% | 0.99+ | 0.32-0.38 |
| DS-Stack Performance | 96-98% | 0.99+ | 0.32-0.36 |
| DS-Stack Combined | 96-98% | 0.99+ | 0.32-0.36 |

---

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, confusion_matrix
)
import numpy as np

# Standard metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# AUC (multi-class)
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=range(4))
auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')

# RMSE (treats classes as ordinal: Good=0, Minor=1, Moderate=2, Critical=3)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

---

## Citation

If using this dataset, please cite the original VEHMS research:

```bibtex
@article{chukwudi2024ensemble,
  title={An Ensemble Deep Learning Model for Vehicular Engine Health Prediction},
  author={Chukwudi, Isinka Joseph and Zaman, Nafees and others},
  journal={IEEE Access},
  volume={12},
  pages={63433--63451},
  year={2024},
  publisher={IEEE}
}
```

---

## Notes

1. **Preprocessing**: Always scale features using `StandardScaler` before training
2. **Stratification**: Use stratified train/test split to maintain class balance
3. **Random Seed**: Use `RANDOM_SEED = 42` for reproducibility
4. **Cross-Validation**: 5-fold CV is recommended for stacking models
5. **Categorical Features**: `Env_Scenario`, `Sensor_Noise_Level`, `Fault_Progression_Stage` need encoding if used
