# VEHMS-DL Implementation Plan

## Comprehensive Analysis Report Implementation Guide

**Document Version:** 1.0  
**Date:** April 25, 2026  
**Based on:** README.md Research Analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Implementation Phases](#2-implementation-phases)
3. [Phase 1: Baseline Reproduction](#3-phase-1-baseline-reproduction)
4. [Phase 2: Deep Learning Integration](#4-phase-2-deep-learning-integration)
5. [Phase 3: Advanced Ensemble Architectures](#5-phase-3-advanced-ensemble-architectures)
6. [Phase 4: Explainability & Uncertainty](#6-phase-4-explainability--uncertainty)
7. [Phase 5: Performance Optimization](#7-phase-5-performance-optimization)
8. [Phase 6: Comparative Analysis](#8-phase-6-comparative-analysis)
9. [Technical Specifications](#9-technical-specifications)
10. [Success Metrics](#10-success-metrics)

---

## 1. Executive Summary

### Objective

Implement and validate the novel contributions outlined in the README.md to achieve:
- **Accuracy:** 97.5-98.5% (current: 96.85%)
- **AUC:** 0.999+ (current: 0.9984)
- **RMSE:** 0.28-0.32 (current: 0.3413)

### Key Innovations to Implement

| Innovation | Priority | Complexity | Expected Impact |
|------------|----------|------------|-----------------|
| True Deep Learning Integration | High | Medium | +1-2% accuracy |
| Multi-Level Heterogeneous Stacking | High | High | +1-3% accuracy |
| Dynamic Classifier Selection Enhancement | Medium | Medium | +0.5-1% accuracy |
| Uncertainty Quantification | Medium | Medium | Confidence intervals |
| Attention-Based Feature Weighting | High | High | +1-2% accuracy |

### Current Codebase Assets

```
vehms/
├── deep_learning_classifiers.py    # CNN, LSTM, GRU, Attention (READY)
├── dynamic_classifier_selector.py  # Q-statistic diversity (READY)
├── stacked_ensemble.py             # Custom stacking (READY)
├── existing_research_stacked_ensemble.py  # ER configs (READY)
├── model_evaluator.py              # Metrics calculation (READY)
├── xai_explainer.py                # SHAP + LIME (READY)
├── base_classifier_module.py       # 10 base classifiers (READY)
├── data_loader.py                  # Data loading (READY)
├── data_preprocessor.py            # Preprocessing (READY)
└── config.py                       # Configuration (READY)
```

---

## 2. Implementation Phases

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION TIMELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Baseline     ████████░░░░░░░░░░░░░░░░░░░░  Week 1        │
│  Phase 2: Deep Learning ░░░░░░░░████████░░░░░░░░░░░░  Week 2        │
│  Phase 3: Ensembles    ░░░░░░░░░░░░░░░░████████░░░░  Week 3        │
│  Phase 4: XAI          ░░░░░░░░░░░░░░░░░░░░░░░░████  Week 4        │
│  Phase 5: Optimization ░░░░░░░░░░░░░░░░░░░░░░░░░░██  Week 4        │
│  Phase 6: Analysis     ░░░░░░░░░░░░░░░░░░░░░░░░░░░█  Week 4        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dependencies

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 5
                │              │
                └──► Phase 4 ──┘
                              │
                              └──► Phase 6
```

---

## 3. Phase 1: Baseline Reproduction

### Objective
Reproduce and validate existing results on both datasets to establish baseline metrics.

### Tasks

#### Task 1.1: Data Pipeline Validation
```python
# Files: vehms/data_loader.py, vehms/data_preprocessor.py

Steps:
1. Load both datasets:
   - dataset/cars1.csv (3,003 samples, 7 features)
   - dataset/augmented_data_with_environment.csv (10,000 samples, 10 features)

2. Validate data integrity:
   - Check for missing values
   - Verify column types
   - Confirm class distribution

3. Preprocess data:
   - StandardScaler for features
   - LabelEncoder for target
   - 80/20 stratified split
```

#### Task 1.2: Base Classifier Training
```python
# File: vehms/base_classifier_module.py

Classifiers to train:
1. LR (Logistic Regression)
2. KNN (K-Nearest Neighbors)
3. LDA (Linear Discriminant Analysis)
4. GNB (Gaussian Naive Bayes)
5. SVM (Support Vector Machine)
6. DT (Decision Tree)
7. RF (Random Forest)
8. AdaBoost
9. GB (Gradient Boosting)
10. XGBoost

Expected Output:
- Individual classifier accuracies
- Training times
- Prediction probabilities for stacking
```

#### Task 1.3: Existing Research Stacking Reproduction
```python
# File: vehms/existing_research_stacked_ensemble.py

Configurations to reproduce:
1. ER-Stacked Model 1: RF + SVM + GB + DT + KNN
2. ER-Stacked Model 2: LR + SVM + LDA + GB + AdaBoost
3. ER-Stacked Model 3: All 9 classifiers

Target Metrics (Original Dataset):
- ER-Stacked Model 3: AUC 0.9725, Accuracy 94.51%
```

#### Task 1.4: Dynamic Classifier Selection Baseline
```python
# File: vehms/dynamic_classifier_selector.py

Selection methods to run:
1. Performance-based (top-k by accuracy)
2. Diversity-based (Q-statistic optimization)
3. Combined (60% performance + 40% diversity)
4. Greedy forward selection

Expected Output:
- Q-statistic diversity matrix
- Selected classifier combinations
- Stacked model performance
```

### Deliverables
- [ ] Baseline metrics table for both datasets
- [ ] Reproducibility verification report
- [ ] Performance comparison with original paper

### Success Criteria
| Metric | Original Dataset Target | Augmented Dataset Target |
|--------|------------------------|-------------------------|
| Accuracy | ≥ 94.50% | ≥ 96.85% |
| AUC | ≥ 0.9725 | ≥ 0.9984 |
| RMSE | ≤ 0.3827 | ≤ 0.3413 |

---

## 4. Phase 2: Deep Learning Integration

### Objective
Integrate true deep learning classifiers into the stacking framework.

### Tasks

#### Task 2.1: Deep Learning Classifier Validation
```python
# File: vehms/deep_learning_classifiers.py

Classifiers to validate:
1. CNNClassifier
   - Architecture: 3 Conv1D blocks + GlobalAvgPool + Dense
   - Parameters: filters=[64,128,256], kernel_size=3
   
2. LSTMClassifier
   - Architecture: Stacked LSTM + Dense
   - Parameters: lstm_units=[64,32], bidirectional=False/True
   
3. GRUClassifier
   - Architecture: Stacked GRU + Dense
   - Parameters: gru_units=[64,32], bidirectional=False/True
   
4. CNNLSTMClassifier
   - Architecture: CNN feature extraction + LSTM sequential
   - Parameters: cnn_filters=[64,128], lstm_units=[64]
   
5. AttentionLSTMClassifier
   - Architecture: LSTM + Attention mechanism
   - Parameters: lstm_units=[64,64], attention_units=64

Training Configuration:
- epochs: 100 (with early stopping, patience=15)
- batch_size: 32
- validation_split: 0.15
- optimizer: Adam (lr=0.001 with ReduceLROnPlateau)
```

#### Task 2.2: Standalone DL Performance Evaluation
```python
# Evaluate each DL classifier independently

Metrics to collect:
- Accuracy, Precision, Recall, F1
- AUC (multi-class OVR)
- RMSE, MAE
- Training time
- Convergence curves

Comparison:
- DL classifiers vs traditional ML classifiers
- Identify best performing DL architecture
```

#### Task 2.3: DL-Enhanced Stacking
```python
# New stacking configurations with DL classifiers

Configuration DL-Stack-1: ML + CNN
- Base: RF, XGBoost, SVM, CNN
- Meta: LogisticRegression

Configuration DL-Stack-2: ML + LSTM
- Base: RF, XGBoost, SVM, LSTM
- Meta: LogisticRegression

Configuration DL-Stack-3: ML + Attention-LSTM
- Base: RF, XGBoost, SVM, AttentionLSTM
- Meta: LogisticRegression

Configuration DL-Stack-4: Full Hybrid
- Base: RF, XGBoost, CNN, LSTM, AttentionLSTM
- Meta: LogisticRegression
```

#### Task 2.4: Hyperparameter Optimization for DL
```python
# Use Optuna or GridSearch for DL hyperparameter tuning

Parameters to tune:
- CNN: filters, kernel_size, dropout_rate
- LSTM: lstm_units, bidirectional, recurrent_dropout
- Attention: attention_units, dropout_rate
- Common: learning_rate, batch_size, epochs
```

### Deliverables
- [ ] DL classifier performance report
- [ ] DL-enhanced stacking configurations
- [ ] Hyperparameter optimization results
- [ ] Training convergence visualizations

### Success Criteria
| Configuration | Target Accuracy | Target AUC |
|---------------|-----------------|------------|
| DL-Stack-1 | ≥ 97.0% | ≥ 0.9985 |
| DL-Stack-2 | ≥ 97.0% | ≥ 0.9985 |
| DL-Stack-3 | ≥ 97.2% | ≥ 0.9988 |
| DL-Stack-4 | ≥ 97.5% | ≥ 0.9990 |

---

## 5. Phase 3: Advanced Ensemble Architectures

### Objective
Implement the novel multi-level heterogeneous stacking architecture.

### Tasks

#### Task 3.1: Multi-Level Stacking Implementation
```python
# New file: vehms/multi_level_stacking.py

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                   │
│  [Sensor Features] + [Environmental Features]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LEVEL 0: HETEROGENEOUS BASE LEARNERS            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  ML BRANCH      │  DL BRANCH      │  PROBABILISTIC BRANCH       │
│  - RF           │  - CNN          │  - GNB                      │
│  - XGBoost      │  - LSTM         │  - LDA                      │
│  - LightGBM     │  - CNN-LSTM     │  - QDA (new)                │
│  - CatBoost     │  - Attention    │                             │
│  - SVM          │    LSTM         │                             │
│  - KNN          │                 │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEVEL 1: BRANCH META-LEARNERS                      │
│  - ML Branch: GradientBoosting                                  │
│  - DL Branch: Dense Network                                     │
│  - Prob Branch: Bayesian Averaging                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEVEL 2: FINAL ENSEMBLE                            │
│  - Attention-Weighted Voting                                    │
│  - Uncertainty Estimation                                       │
└─────────────────────────────────────────────────────────────────┘

Implementation Steps:
1. Create BranchEnsemble class for each branch
2. Create MultiLevelStackingClassifier class
3. Implement attention-weighted voting mechanism
4. Add uncertainty estimation layer
```

#### Task 3.2: Branch-Specific Meta-Learners
```python
# ML Branch Meta-Learner
ml_meta = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

# DL Branch Meta-Learner
dl_meta = Sequential([
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])

# Probabilistic Branch Meta-Learner
prob_meta = BayesianAveraging()  # Custom implementation
```

#### Task 3.3: Attention-Weighted Voting
```python
# New file: vehms/attention_voting.py

class AttentionWeightedVoting:
    """
    Learn attention weights for combining branch predictions.
    Weights can be:
    - Static (learned during training)
    - Dynamic (context-dependent based on input features)
    """
    
    def __init__(self, n_branches, context_aware=True):
        self.n_branches = n_branches
        self.context_aware = context_aware
        
    def compute_weights(self, X, branch_predictions):
        if self.context_aware:
            # Use environmental features to adjust weights
            # e.g., different weights for high altitude vs low altitude
            pass
        else:
            # Use learned static weights
            pass
```

#### Task 3.4: Enhanced Dynamic Classifier Selection
```python
# Enhance vehms/dynamic_classifier_selector.py

New selection methods:
1. Correlation-based diversity (complement Q-statistic)
2. Error pattern analysis
3. Confidence-weighted selection
4. Cross-validation stability selection

New features:
- Automatic branch assignment based on classifier type
- Diversity optimization within and across branches
- Performance-diversity Pareto frontier analysis
```

### Deliverables
- [ ] MultiLevelStackingClassifier implementation
- [ ] AttentionWeightedVoting implementation
- [ ] Enhanced DynamicClassifierSelector
- [ ] Architecture comparison report

### Success Criteria
| Architecture | Target Accuracy | Target AUC |
|--------------|-----------------|------------|
| 2-Level Stack | ≥ 97.3% | ≥ 0.9988 |
| 3-Level Stack | ≥ 97.8% | ≥ 0.9992 |
| Attention Voting | ≥ 98.0% | ≥ 0.9995 |

---

## 6. Phase 4: Explainability & Uncertainty

### Objective
Implement comprehensive explainability and uncertainty quantification.

### Tasks

#### Task 4.1: Enhanced SHAP Analysis
```python
# Enhance vehms/xai_explainer.py

New SHAP features:
1. TreeExplainer for tree-based models (faster)
2. DeepExplainer for DL models
3. Ensemble SHAP (aggregate explanations)
4. Temporal SHAP for sequential models

Visualizations:
- Global feature importance (bar plot)
- Local explanations (force plot, waterfall)
- Feature interactions (dependence plots)
- Class-specific importance
```

#### Task 4.2: Attention Weight Visualization
```python
# For AttentionLSTMClassifier

Features:
1. Extract attention weights per sample
2. Visualize attention heatmaps
3. Aggregate attention patterns by class
4. Compare attention vs SHAP importance
```

#### Task 4.3: Uncertainty Quantification
```python
# New file: vehms/uncertainty_estimator.py

class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using multiple methods.
    """
    
    Methods:
    1. Monte Carlo Dropout
       - Enable dropout at inference
       - Run multiple forward passes
       - Compute mean and variance
       
    2. Ensemble Disagreement
       - Measure variance across ensemble members
       - Higher disagreement = higher uncertainty
       
    3. Prediction Entropy
       - H = -Σ p(y) log p(y)
       - Higher entropy = higher uncertainty
       
    4. Calibration Analysis
       - Expected Calibration Error (ECE)
       - Reliability diagrams
```

#### Task 4.4: Confidence Intervals
```python
# Add confidence intervals to predictions

Output format:
{
    'prediction': 'Critical',
    'confidence': 0.92,
    'uncertainty': 'Low',
    'confidence_interval': [0.88, 0.96],
    'feature_contributions': {...}
}
```

### Deliverables
- [ ] Enhanced XAI module
- [ ] Uncertainty estimation module
- [ ] Attention visualization tools
- [ ] Calibration analysis report

### Success Criteria
- SHAP explanations for all model types
- Uncertainty estimates with <5% calibration error
- Attention weights correlate with SHAP importance (r > 0.7)

---

## 7. Phase 5: Performance Optimization

### Objective
Optimize models to achieve target performance metrics.

### Tasks

#### Task 5.1: Hyperparameter Tuning
```python
# Use Optuna for Bayesian optimization

Parameters to tune:
1. Base classifiers:
   - RF: n_estimators, max_depth, min_samples_split
   - XGBoost: n_estimators, max_depth, learning_rate, subsample
   - SVM: C, gamma, kernel
   
2. Deep learning:
   - Architecture: layers, units, dropout
   - Training: learning_rate, batch_size, epochs
   
3. Stacking:
   - CV folds
   - Meta-learner parameters
   - Passthrough option

Optimization strategy:
- 100 trials per model
- 5-fold cross-validation
- Maximize AUC as primary metric
```

#### Task 5.2: Feature Engineering
```python
# New features to create

1. Interaction features:
   - Crankshaft × Overheating
   - Lubricant × Piston
   - Temperature × Humidity
   
2. Polynomial features:
   - Degree 2 for top features
   
3. Domain-specific features:
   - Engine stress index
   - Environmental severity score
   - Maintenance urgency indicator
   
4. Feature selection:
   - Remove low-importance features
   - Keep top-k by SHAP importance
```

#### Task 5.3: Ensemble Weight Optimization
```python
# Optimize ensemble combination weights

Methods:
1. Grid search over weight combinations
2. Gradient-based optimization
3. Evolutionary algorithms (CMA-ES)
4. Stacking with learned weights
```

#### Task 5.4: Cross-Validation Strategy
```python
# Compare CV strategies

Strategies to test:
1. StratifiedKFold (k=5, 10)
2. RepeatedStratifiedKFold (5×3, 5×5)
3. Nested CV for unbiased evaluation
4. Time-based split (if temporal patterns exist)
```

### Deliverables
- [ ] Optimized hyperparameters for all models
- [ ] Feature engineering pipeline
- [ ] Ensemble weight optimization results
- [ ] CV strategy comparison

### Success Criteria
| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Accuracy | 96.85% | 97.5% | 98.5% |
| AUC | 0.9984 | 0.999 | 0.9995 |
| RMSE | 0.3413 | 0.32 | 0.28 |

---

## 8. Phase 6: Comparative Analysis

### Objective
Comprehensive comparison and documentation for research publication.

### Tasks

#### Task 6.1: Original Paper Comparison
```python
# Compare against Chukwudi et al. (2024)

Comparison dimensions:
1. Dataset characteristics
   - Size: 3,003 vs 10,000 samples
   - Features: 6 vs 9 features
   
2. Methodology
   - "Deep learning" claim vs actual implementation
   - Static vs dynamic ensemble selection
   
3. Performance metrics
   - All metrics on original dataset
   - Improvement quantification
```

#### Task 6.2: Ablation Study
```python
# Systematic ablation to quantify contributions

Ablations:
1. Remove environmental features
2. Remove DL classifiers
3. Remove attention mechanism
4. Remove uncertainty estimation
5. Use static vs dynamic selection
6. Single-level vs multi-level stacking

Report:
- Performance impact of each component
- Statistical significance (paired t-test)
```

#### Task 6.3: Statistical Significance Testing
```python
# Rigorous statistical analysis

Tests:
1. Paired t-test for accuracy differences
2. McNemar's test for classifier comparison
3. Friedman test for multiple classifier comparison
4. Nemenyi post-hoc test
5. Confidence intervals via bootstrap
```

#### Task 6.4: Visualization Suite
```python
# Comprehensive visualizations for paper

Figures to generate:
1. Architecture diagram (multi-level stacking)
2. Performance comparison bar charts
3. ROC curves (multi-class)
4. Confusion matrices (heatmaps)
5. Feature importance (SHAP summary)
6. Attention weight heatmaps
7. Uncertainty calibration plots
8. Learning curves
9. Diversity matrix heatmap
10. Ablation study results
```

#### Task 6.5: Research Paper Preparation
```markdown
# Paper outline

1. Abstract
2. Introduction
   - Problem statement
   - Limitations of existing work
   - Contributions
3. Related Work
   - VEHMS systems
   - Ensemble methods
   - Deep learning for predictive maintenance
4. Methodology
   - Dataset description
   - Proposed architecture
   - Training procedure
5. Experiments
   - Experimental setup
   - Baseline comparison
   - Ablation study
   - Statistical analysis
6. Results and Discussion
7. Conclusion and Future Work
```

### Deliverables
- [ ] Comprehensive comparison report
- [ ] Ablation study results
- [ ] Statistical significance analysis
- [ ] Publication-ready figures
- [ ] Paper draft outline

### Success Criteria
- Demonstrate statistically significant improvement (p < 0.05)
- Clear ablation showing contribution of each component
- Publication-quality visualizations

---

## 9. Technical Specifications

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | - | NVIDIA GTX 1080+ |
| Storage | 10 GB | 50 GB SSD |

### Software Dependencies

```python
# Core ML
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Deep Learning
tensorflow>=2.8.0
keras>=2.8.0

# Gradient Boosting
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Explainability
shap>=0.40.0
lime>=0.2.0

# Hyperparameter Tuning
optuna>=2.10.0

# Statistical Testing
scipy>=1.7.0
statsmodels>=0.13.0
```

### Code Organization

```
vehms/
├── __init__.py
├── config.py                          # Configuration
├── data_loader.py                     # Data loading
├── data_preprocessor.py               # Preprocessing
├── data_cleaner.py                    # Data cleaning
├── data_visualizer.py                 # Visualization
├── feature_extractor.py               # Feature engineering
├── base_classifier_module.py          # Base classifiers
├── deep_learning_classifiers.py       # DL classifiers
├── stacked_ensemble.py                # Custom stacking
├── existing_research_stacked_ensemble.py  # ER configs
├── dynamic_classifier_selector.py     # Dynamic selection
├── multi_level_stacking.py            # NEW: Multi-level arch
├── attention_voting.py                # NEW: Attention voting
├── uncertainty_estimator.py           # NEW: Uncertainty
├── model_evaluator.py                 # Evaluation
├── performance_visualizer.py          # Performance viz
├── xai_explainer.py                   # SHAP + LIME
└── hyperparameter_tuner.py            # Optuna tuning
```

---

## 10. Success Metrics

### Primary Metrics

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Accuracy | 96.85% | 97.5% | 98.5% |
| AUC | 0.9984 | 0.999 | 0.9995 |
| RMSE | 0.3413 | 0.32 | 0.28 |
| Precision | 96.85% | 97.5% | 98.5% |
| F1-Score | 96.80% | 97.5% | 98.5% |

### Secondary Metrics

| Metric | Target |
|--------|--------|
| Training Time | < 10 minutes |
| Inference Time | < 100ms per sample |
| Model Size | < 500 MB |
| Calibration Error | < 5% |

### Research Impact Metrics

| Metric | Target |
|--------|--------|
| Improvement over original paper | > 2% accuracy |
| Statistical significance | p < 0.05 |
| Novel contributions | ≥ 3 |
| Ablation components | ≥ 5 |

---

## Appendix A: Quick Reference Commands

### Phase 1: Baseline
```python
# Load data
from vehms import DataLoader, DataPreprocessor
loader = DataLoader()
df = loader.load_dataset('dataset/augmented_data_with_environment.csv')

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

# Train base classifiers
from vehms import BaseClassifierModule
base_module = BaseClassifierModule()
results_df = base_module.train_all(X_train, y_train, X_test, y_test)
```

### Phase 2: Deep Learning
```python
# Train DL classifiers
from vehms.deep_learning_classifiers import CNNClassifier, LSTMClassifier, AttentionLSTMClassifier

cnn = CNNClassifier(n_features=9, n_classes=4, epochs=100, verbose=1)
cnn.fit(X_train, y_train)

lstm = LSTMClassifier(n_features=9, n_classes=4, bidirectional=True)
lstm.fit(X_train, y_train)

attention = AttentionLSTMClassifier(n_features=9, n_classes=4)
attention.fit(X_train, y_train)
```

### Phase 3: Advanced Ensembles
```python
# Dynamic classifier selection
from vehms import DynamicClassifierSelector
selector = DynamicClassifierSelector(cv=5, random_state=42)
results = selector.run_all_selection_methods(X_train, y_train, X_test, y_test, top_k=5)

# Create dynamic stack
dynamic_stack = selector.create_dynamic_stack(results['combined'])
dynamic_stack.fit(X_train, y_train)
```

### Phase 4: Explainability
```python
# XAI analysis
from vehms import XAIExplainer
explainer = XAIExplainer(model, feature_names, class_names)
explainer.initialize_shap(X_train)
explainer.compute_shap_values(X_test[:100])
explainer.plot_shap_summary()

explainer.initialize_lime(X_train)
explanation = explainer.explain_instance_lime(X_test[0])
explainer.plot_lime_explanation(explanation)
```

### Phase 5: Evaluation
```python
# Comprehensive evaluation
from vehms import ModelEvaluator
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test, 'Model Name')
comparison_df = evaluator.compare_models()
```

---

## Appendix B: Checkpoint Milestones

| Checkpoint | Phase | Deliverable | Due |
|------------|-------|-------------|-----|
| CP1 | 1 | Baseline metrics validated | Week 1 |
| CP2 | 2 | DL classifiers integrated | Week 2 |
| CP3 | 3 | Multi-level stacking working | Week 3 |
| CP4 | 4 | XAI + Uncertainty complete | Week 4 |
| CP5 | 5 | Performance targets achieved | Week 4 |
| CP6 | 6 | Analysis report complete | Week 4 |

---

## Appendix C: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DL overfitting | Medium | High | Early stopping, dropout, regularization |
| Computational cost | Medium | Medium | GPU acceleration, batch processing |
| Target not achieved | Low | High | Iterative optimization, ensemble diversity |
| Reproducibility issues | Low | Medium | Fixed random seeds, version control |

---

**Document End**

*Last Updated: April 25, 2026*
