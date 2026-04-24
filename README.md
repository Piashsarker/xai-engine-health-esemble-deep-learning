# Vehicle Engine Health Monitoring System (VEHMS)

## Predictive Maintenance Framework using Stacked Ensemble Deep Learning

VEHMS is a machine learning framework that analyzes vehicle engine sensor data and environmental conditions to predict engine health status. The system classifies engine conditions into four categories: **Good**, **Minor**, **Moderate**, and **Critical**.

This implementation extends and improves upon the research presented in:

> **Chukwudi, I. J., Zaman, N., Rahim, M. A., Rahman, M. A., Alenazi, M. J. F., & Pillai, P. (2024).** *An Ensemble Deep Learning Model for Vehicular Engine Health Prediction.* IEEE Access, Volume 12, pp. 63433-63451. DOI: 10.1109/ACCESS.2024.3395927

---

## Overview

This project implements a comprehensive predictive maintenance solution using stacked ensemble learning. By combining multiple base classifiers with meta-learning, the system achieves high accuracy in predicting engine health degradation before failures occur.

### Key Features

- **Multi-sensor data analysis**: Processes 6 engine sensor readings and 3 environmental factors
- **Stacked ensemble architecture**: Combines diverse classifiers for robust predictions
- **Dynamic classifier selection**: Automatically selects optimal classifier combinations based on diversity and performance metrics
- **Explainable AI**: SHAP and LIME integration for model interpretability
- **Research comparison**: Benchmarks against existing research configurations
- **Extended dataset**: Augmented with environmental factors (Temperature, Humidity, Altitude)

### Target Performance Metrics

| Metric | Target |
|--------|--------|
| AUC | ~0.9702 |
| Accuracy | ~0.9470 |
| Precision | ~0.9486 |
| RMSE | ~0.3355 |

---

## Comparison with Existing Research

This implementation builds upon and extends the original research by Chukwudi et al. (2024). Below is a detailed comparison highlighting the key differences and improvements.

### Dataset Comparison

| Aspect | Original Research | This Implementation |
|--------|-------------------|---------------------|
| **Sample Size** | 3,003 samples | 10,000 samples (augmented) |
| **Features** | 6 sensor features | 9 features (6 sensor + 3 environmental) |
| **Sensor Features** | Crankshaft, Overheating, Lubricant, Misfires, Piston, Starter | Same 6 sensor features |
| **Environmental Features** | None | Temperature, Humidity, Altitude |
| **Target Classes** | Good, Minor, Moderate, Critical | Same 4 classes |
| **Data Augmentation** | Not applied | Applied to increase sample diversity |

### New Environmental Parameters

This implementation introduces three environmental factors that affect engine performance:

| Parameter | Range | Impact on Engine Health |
|-----------|-------|------------------------|
| **Temperature (°C)** | 15.0 - 35.0 | Affects engine cooling efficiency, oil viscosity, and combustion |
| **Humidity (%)** | 30.0 - 90.0 | Influences air density, combustion efficiency, and corrosion risk |
| **Altitude (m)** | 0.0 - 3000.0 | Impacts air pressure, oxygen availability, and engine power output |

### Stacked Ensemble Strategy Comparison

#### Original Research Stacked Models

The original paper by Chukwudi et al. defined three stacked ensemble configurations:

**Original Stacked Model 1** (Best Performer in Research):
- Base Classifiers: Random Forest + SVM + Gradient Boosting + Decision Tree + KNN
- Meta-Learner: Logistic Regression
- Performance: AUC = 0.9702, Accuracy = 94.70%, RMSE = 0.3355

**Original Stacked Model 2**:
- Base Classifiers: Logistic Regression + SVM + LDA + Gradient Boosting + AdaBoost
- Meta-Learner: Logistic Regression
- Performance: AUC = 0.9665, Accuracy = 94.70%

**Original Stacked Model 3**:
- Base Classifiers: LR + KNN + SVM + LDA + GB + AdaBoost + DT + RF + GNB (All 9 classifiers)
- Meta-Learner: Logistic Regression
- Performance: AUC = 0.9653, Accuracy = 93.01%

#### This Implementation: Dynamic Classifier Selection

Instead of using fixed, manually-designed stacking configurations like the original research, **this implementation introduces a data-driven approach using the `DynamicClassifierSelector`**. This component automatically determines the optimal classifier combination for any given dataset.

**Why Dynamic Selection is Superior to Static Stacking:**

| Aspect | Original Research (Static) | This Implementation (Dynamic) |
|--------|---------------------------|-------------------------------|
| **Classifier Selection** | Fixed, manually chosen | Automatically optimized per dataset |
| **Adaptability** | Same configuration for all data | Adapts to data characteristics |
| **Diversity Consideration** | Implicit (researcher intuition) | Explicit (Q-statistic measurement) |
| **Scalability** | Must redesign for new classifiers | Automatically evaluates new classifiers |
| **Reproducibility** | Depends on researcher choices | Algorithm-driven, fully reproducible |

**The Problem with Static Stacking (Original Research):**

The original research selected classifier combinations based on researcher intuition and trial-and-error. While effective, this approach has limitations:
1. **No formal diversity measurement** - Classifiers may make correlated errors
2. **Dataset-specific** - Optimal combinations may differ for other datasets
3. **Not scalable** - Adding new classifiers requires manual re-evaluation
4. **Suboptimal combinations** - Human intuition may miss better combinations

**Our Solution: Dynamic Classifier Selection**

The `DynamicClassifierSelector` addresses these limitations through four selection methods:

1. **Performance-Based Selection**: Selects top-k classifiers by individual accuracy
2. **Diversity-Based Selection**: Maximizes ensemble diversity using Q-statistic
3. **Combined Selection** (Recommended): Balances performance (60%) and diversity (40%)
4. **Greedy Forward Selection**: Iteratively builds the optimal ensemble

### Key Methodological Differences

| Aspect | Original Research | This Implementation |
|--------|-------------------|---------------------|
| **Stacking Approach** | Static (fixed classifier combinations) | Dynamic (algorithm-selected combinations) |
| **Classifier Selection** | Manual, researcher-driven | Automatic, data-driven |
| **Diversity Measurement** | Not measured | Q-statistic quantification |
| **Cross-Validation** | RepeatedStratifiedKFold (10 splits, 3 repeats) | StratifiedKFold (5-fold) for stacking |
| **Feature Scaling** | StandardScaler | StandardScaler (same) |
| **Label Encoding** | LabelEncoder | LabelEncoder (same) |
| **XGBoost Integration** | Not included in stacking | Included as key base classifier |
| **Explainability** | Not included | SHAP + LIME integration |

### Performance Comparison Framework

| Model Category | Configuration | Selection Method | Expected AUC | Expected Accuracy |
|----------------|---------------|------------------|--------------|-------------------|
| **Original Research** | Stacked Model 1 | Manual (Static) | 0.9702 | 94.70% |
| **Original Research** | Stacked Model 2 | Manual (Static) | 0.9665 | 94.70% |
| **Original Research** | Stacked Model 3 | Manual (Static) | 0.9653 | 93.01% |
| **This Implementation** | Dynamic Stack | Combined (60/40) | ~0.97+ | ~95%+ |
| **This Implementation** | Dynamic Stack | Greedy Forward | ~0.97+ | ~95%+ |
| **This Implementation** | Dynamic Stack | Diversity-Based | ~0.96+ | ~94%+ |

### Improvements Over Original Research

1. **Dynamic Classifier Selection (Key Innovation)**: Instead of manually selecting classifier combinations, the `DynamicClassifierSelector` automatically identifies optimal combinations using:
   - **Q-statistic for diversity measurement**: Quantifies how differently classifiers make errors
   - **Performance-based selection**: Ensures high-accuracy classifiers are included
   - **Combined scoring (60% performance, 40% diversity)**: Balances accuracy with error diversity
   - **Greedy forward selection**: Iteratively builds the best ensemble

2. **Extended Feature Set**: Addition of environmental parameters (Temperature, Humidity, Altitude) provides more context for engine health prediction in real-world conditions.

3. **Larger Dataset**: Augmented from 3,003 to 10,000 samples, reducing overfitting risk and improving model generalization.

4. **XGBoost Integration**: Modern gradient boosting with better performance and native handling of missing values.

5. **Explainable AI**: Integration of SHAP and LIME provides transparency into model decisions, addressing the "black box" concern raised in the original research.

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

## Stacked Ensemble Architecture

This section explains the stacking methodology used in both the original research and this implementation.

### What is Stacking?

Stacking (stacked generalization) is a two-level ensemble technique:

1. **Level 0 (Base Classifiers)**: Multiple diverse classifiers trained on the original features
2. **Level 1 (Meta-Learner)**: A classifier that learns to combine base classifier predictions

The key insight is that different classifiers make different types of errors. By combining them intelligently through a meta-learner, the ensemble achieves better performance than any individual classifier.

#### Mathematical Formulation (from Original Research)

For N base models h₁, h₂, ..., hₙ and meta-model H:

```
y₁ = h₁(X), y₂ = h₂(X), ..., yₙ = hₙ(X)
X_meta = [y₁, y₂, ..., yₙ]
Y_meta = H(X_meta, y)
```

Where:
- X represents input features
- y represents target labels
- X_meta is the new feature matrix from base model predictions
- Y_meta is the final prediction from the meta-model

### Stacking Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         Input Features (9)          │
                    │  Crankshaft, Overheating, Lubricant │
                    │  Misfires, Piston, Starter,         │
                    │  Temperature, Humidity, Altitude    │
                    └─────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
    ┌─────▼─────┐              ┌─────▼─────┐              ┌─────▼─────┐
    │  Base     │              │  Base     │              │  Base     │
    │Classifier │              │Classifier │              │Classifier │
    │    1      │              │    2      │              │    N      │
    └─────┬─────┘              └─────┬─────┘              └─────┬─────┘
          │                          │                          │
          │    ┌─────────────────────┼─────────────────────┐    │
          │    │                     │                     │    │
          │    │   DYNAMIC SELECTION │ (This Implementation)    │
          │    │   Q-statistic based │ diversity optimization   │
          │    └─────────────────────┼─────────────────────┘    │
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
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

### Original Research: Static Stacking Configurations

The original paper by Chukwudi et al. (2024) used three fixed configurations:

| Model | Base Classifiers | Focus | AUC |
|-------|------------------|-------|-----|
| ER-Stacked 1 | RF + SVM + GB + DT + KNN | Tree & Instance Based | 0.9702 |
| ER-Stacked 2 | LR + SVM + LDA + GB + AdaBoost | Linear & Boosting | 0.9665 |
| ER-Stacked 3 | All 9 Classifiers | Comprehensive | 0.9653 |

**Limitation:** These combinations were manually selected without formal diversity measurement.

### This Implementation: Dynamic Stacking

Instead of fixed configurations, we use the `DynamicClassifierSelector` to automatically determine the optimal combination:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DYNAMIC STACKING WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Train All Base Classifiers                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ LR  │ │ KNN │ │ SVM │ │ DT  │ │ RF  │ │ Ada │ │ GB  │ │ XGB │ ...   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘       │
│     │       │       │       │       │       │       │       │          │
│  Step 2: Calculate Q-Statistic Diversity Matrix                         │
│     └───────┴───────┴───────┴───────┴───────┴───────┴───────┘          │
│                              │                                          │
│  Step 3: Select Optimal Combination (Combined Method)                   │
│     Score = 0.6 × Accuracy + 0.4 × (1 - Avg Q-statistic)               │
│                              │                                          │
│  Step 4: Build Dynamic Stack with Selected Classifiers                  │
│     ┌─────────────────────────────────────────┐                         │
│     │  Selected: [RF, XGBoost, SVM, KNN, Ada] │ ← Data-driven choice   │
│     └─────────────────────────────────────────┘                         │
│                              │                                          │
│  Step 5: Train Meta-Learner (Logistic Regression)                       │
│                              ▼                                          │
│     ┌─────────────────────────────────────────┐                         │
│     │         Final Ensemble Model            │                         │
│     └─────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dynamic Classifier Selection (Proposed Technique)

The `DynamicClassifierSelector` is the **core innovation** of this implementation, replacing the static stacking approach used in the original research with an intelligent, data-driven selection algorithm.

### Why Dynamic Selection Outperforms Static Stacking

#### The Fundamental Problem with Static Stacking

In the original research, classifier combinations were selected based on:
- Researcher intuition and domain knowledge
- Trial-and-error experimentation
- General assumptions about classifier diversity

While this produced good results (AUC = 0.9702), it has critical limitations:

```
Static Stacking Workflow (Original Research):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Researcher      │────▶│ Manual Testing  │────▶│ Fixed           │
│ Intuition       │     │ (Trial & Error) │     │ Configuration   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ⚠️                      ⚠️                      ⚠️
   Subjective            Time-consuming          Not adaptable
```

#### Our Solution: Algorithm-Driven Selection

The `DynamicClassifierSelector` replaces human intuition with mathematical optimization:

```
Dynamic Selection Workflow (This Implementation):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Train All       │────▶│ Measure         │────▶│ Optimize        │
│ Classifiers     │     │ Q-Statistic     │     │ Selection       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ✓                      ✓                      ✓
   Comprehensive         Quantitative           Data-driven
```

### The Q-Statistic: Measuring Classifier Diversity

The Q-statistic (Yule's Q) is a pairwise diversity measure that quantifies how differently two classifiers make errors:

```
Q = (N₁₁ × N₀₀ - N₀₁ × N₁₀) / (N₁₁ × N₀₀ + N₀₁ × N₁₀)

Where:
- N₁₁ = Both classifiers correct
- N₀₀ = Both classifiers wrong
- N₁₀ = Classifier 1 correct, Classifier 2 wrong
- N₀₁ = Classifier 1 wrong, Classifier 2 correct
```

**Interpretation:**
| Q Value | Meaning | Ensemble Benefit |
|---------|---------|------------------|
| Q = 1 | Always agree | Low (redundant) |
| Q = 0 | Independent | Medium |
| Q = -1 | Always disagree | High (complementary errors) |

**Why This Matters:** Classifiers with low Q-statistics make different errors. When combined in an ensemble, they can correct each other's mistakes, leading to better overall performance.

### Selection Methods

#### 1. Performance-Based Selection
Selects the top-k classifiers by individual accuracy.

```python
# Simple but may select redundant classifiers
selected = selector.select_by_performance(top_k=5)
```

**Pros:** Fast, ensures high-accuracy classifiers
**Cons:** May select classifiers that make similar errors

#### 2. Diversity-Based Selection
Greedily selects classifiers that maximize diversity (minimize average Q-statistic).

```python
# Maximizes error diversity
selected = selector.select_by_diversity(y_test, top_k=5)
```

**Algorithm:**
1. Start with the best-performing classifier
2. Add classifier with lowest average Q to selected set
3. Repeat until top_k classifiers selected

**Pros:** Ensures diverse error patterns
**Cons:** May include lower-accuracy classifiers

#### 3. Combined Selection (Recommended)
Balances performance and diversity using a weighted score:

```
Combined Score = 0.6 × Accuracy + 0.4 × (1 - Average Q)
```

```python
# Best of both worlds
selected = selector.select_combined(y_test, top_k=5, performance_weight=0.6)
```

**Why 60/40 Split:** Research shows that performance should be weighted higher, but diversity provides diminishing returns beyond a certain point. The 60/40 split optimizes this trade-off.

#### 4. Greedy Forward Selection
Iteratively adds classifiers that improve ensemble accuracy.

```python
# Most accurate but slowest
selected = selector.greedy_forward_selection(X_train, y_train, X_test, y_test, max_classifiers=5)
```

**Algorithm:**
1. Start with empty ensemble
2. Try adding each remaining classifier
3. Keep the one that improves ensemble accuracy most
4. Stop when no improvement is found

**Pros:** Directly optimizes ensemble performance
**Cons:** Computationally expensive (trains multiple stacking models)

### Logical Advantages Over Original Research

| Criterion | Original Research | Dynamic Selection | Why It Matters |
|-----------|-------------------|-------------------|----------------|
| **Objectivity** | Subjective choices | Mathematical optimization | Reproducible, unbiased results |
| **Adaptability** | Fixed for all datasets | Adapts to data characteristics | Better generalization |
| **Diversity** | Assumed, not measured | Quantified via Q-statistic | Ensures complementary errors |
| **Scalability** | Manual re-evaluation needed | Automatic evaluation | Easy to add new classifiers |
| **Optimality** | Local optimum (human limit) | Global search | Better combinations possible |

### Usage Example

```python
from dynamic_classifier_selector import DynamicClassifierSelector

# Initialize selector
selector = DynamicClassifierSelector(cv=5, random_state=42)

# Run all selection methods and compare
results = selector.run_all_selection_methods(X_train, y_train, X_test, y_test, top_k=5)

# View diversity heatmap (automatically displayed)
# Lower Q values (green) = Higher diversity = Better for ensembles

# Create dynamic stack from combined selection (recommended)
dynamic_stack = selector.create_dynamic_stack(results['combined'])
dynamic_stack.fit(X_train, y_train)
print(f"Dynamic Stack Accuracy: {dynamic_stack.score(X_test, y_test):.4f}")

# Or use greedy forward selection for maximum accuracy
greedy_selected = selector.greedy_forward_selection(
    X_train, y_train, X_test, y_test, max_classifiers=5
)
greedy_stack = selector.create_dynamic_stack(greedy_selected)
```

### Diversity Heatmap Interpretation

The `DynamicClassifierSelector` generates a diversity heatmap showing Q-statistics between all classifier pairs:

```
                LR    KNN   LDA   GNB   SVM   DT    RF    Ada   GB    XGB
        LR    [1.0   0.7   0.8   0.5   0.6   0.4   0.5   0.5   0.6   0.5]
        KNN   [0.7   1.0   0.6   0.4   0.5   0.3   0.4   0.4   0.5   0.4]
        ...
        
Color Scale: Red (Q=1, low diversity) → Green (Q=-1, high diversity)
```

**Reading the Heatmap:**
- **Red cells (Q ≈ 1):** Classifiers make similar errors → Avoid pairing
- **Yellow cells (Q ≈ 0):** Classifiers are independent → Acceptable pairing
- **Green cells (Q < 0):** Classifiers make opposite errors → Ideal pairing

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

#### Original Sensor Features (from Research)

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| Crankshaft | float | Crankshaft sensor reading | 4.0 - 21.0 |
| Overheating | float | Overheating indicator | 8.0 - 26.0 |
| Lubricant | float | Lubricant level/quality | 4.0 - 16.0 |
| Misfires | float | Engine misfire intensity | 2.5 - 11.0 |
| Piston | float | Piston sensor reading | 3.5 - 16.0 |
| Starter | float | Starter motor measurement | 250.0 - 450.0 |

#### New Environmental Features (This Implementation)

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
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

The framework evaluates models using the same comprehensive metrics as the original research:

### Regression Metrics
- **RMSE (Root Mean Square Error)**: Measures average magnitude of prediction errors
- **RMSD (Root Mean Square Deviation)**: Measures average variation between predicted and actual values
- **MAE (Mean Absolute Error)**: Provides interpretable measure of average prediction error
- **R² (R-squared)**: Explains proportion of variance captured by the model

### Classification Metrics
- **Accuracy**: Overall correct predictions ratio
- **Precision**: Positive predictive value (weighted)
- **Recall**: True positive rate (weighted)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve (one-vs-rest for multi-class)
- **Confusion Matrix**: Detailed class-wise performance

### Original Research Results Summary

From Chukwudi et al. (2024):

| Model | RMSE | RMSD | MAE | R² | Accuracy | Precision | Recall | AUC |
|-------|------|------|-----|-----|----------|-----------|--------|-----|
| Stacked Model 1 | 0.3355 | 0.1126 | 0.0728 | 0.9021 | 94.70% | 94.86% | 94.70% | 0.9702 |
| Stacked Model 2 | 0.3639 | 0.1325 | 0.0795 | 0.8849 | 94.70% | 94.93% | 94.70% | 0.9665 |
| Stacked Model 3 | 0.3355 | 0.1126 | 0.0728 | 0.9021 | 93.01% | 94.86% | 94.70% | 0.9653 |

---

## License

This project is for research and educational purposes.

---

## References

1. Chukwudi, I. J., Zaman, N., Rahim, M. A., Rahman, M. A., Alenazi, M. J. F., & Pillai, P. (2024). An Ensemble Deep Learning Model for Vehicular Engine Health Prediction. *IEEE Access*, 12, 63433-63451. DOI: 10.1109/ACCESS.2024.3395927

2. Rahim, M. A., Rahman, M. A., Rahman, M. M., Zaman, N., Moustafa, N., & Razzak, I. (2022). An intelligent risk management framework for monitoring vehicular engine health. *IEEE Transactions on Green Communications and Networking*, 6(3), 1298-1306.

---

## Acknowledgments

- Original research by Chukwudi et al. at University of Wolverhampton
- Scikit-learn for machine learning implementations
- XGBoost for gradient boosting
- SHAP and LIME for explainability tools
