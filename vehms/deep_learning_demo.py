# ============================================================
# Deep Learning Classifiers Demo
# ============================================================
# Purpose: Demonstrate usage of CNN, LSTM, and hybrid classifiers
# ============================================================

"""
VEHMS Deep Learning Classifiers - Usage Examples

This module provides example code for using the deep learning classifiers
in the VEHMS predictive maintenance system.

Copy and paste these examples into your Jupyter notebook to get started.
"""


def demo_individual_classifiers():
    """
    Example: Using individual deep learning classifiers.
    
    Copy this code into a notebook cell:
    """
    code = '''
# ============================================================
# DEEP LEARNING CLASSIFIERS - INDIVIDUAL MODELS
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import VEHMS modules
from vehms import (
    DataLoader, DataPreprocessor, ModelEvaluator,
    CNNClassifier, LSTMClassifier, CNNLSTMClassifier,
    GRUClassifier, AttentionLSTMClassifier, RANDOM_SEED
)

# Set random seed
np.random.seed(RANDOM_SEED)

# Load and preprocess data
loader = DataLoader()
df = loader.load_dataset('dataset/augmented_data_with_environment.csv')

preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

# Initialize evaluator
evaluator = ModelEvaluator()

print("\\n" + "="*60)
print("TRAINING DEEP LEARNING CLASSIFIERS")
print("="*60)

# 1. CNN Classifier
print("\\n[1] Training CNN Classifier...")
cnn = CNNClassifier(
    filters=[64, 128, 256],
    dropout_rate=0.3,
    epochs=50,
    batch_size=32,
    verbose=0
)
cnn.fit(X_train, y_train)
cnn_metrics = evaluator.evaluate_model(cnn, X_test, y_test, 'CNN')
print(f"   CNN Test Accuracy: {cnn_metrics['Accuracy']:.4f}")

# 2. LSTM Classifier
print("\\n[2] Training LSTM Classifier...")
lstm = LSTMClassifier(
    lstm_units=[64, 32],
    dropout_rate=0.3,
    epochs=50,
    batch_size=32,
    verbose=0
)
lstm.fit(X_train, y_train)
lstm_metrics = evaluator.evaluate_model(lstm, X_test, y_test, 'LSTM')
print(f"   LSTM Test Accuracy: {lstm_metrics['Accuracy']:.4f}")

# 3. Bidirectional LSTM
print("\\n[3] Training Bidirectional LSTM...")
bilstm = LSTMClassifier(
    lstm_units=[64, 32],
    bidirectional=True,
    dropout_rate=0.3,
    epochs=50,
    batch_size=32,
    verbose=0
)
bilstm.fit(X_train, y_train)
bilstm_metrics = evaluator.evaluate_model(bilstm, X_test, y_test, 'BiLSTM')
print(f"   BiLSTM Test Accuracy: {bilstm_metrics['Accuracy']:.4f}")

# 4. CNN-LSTM Hybrid
print("\\n[4] Training CNN-LSTM Hybrid...")
cnn_lstm = CNNLSTMClassifier(
    cnn_filters=[64, 128],
    lstm_units=[64],
    dropout_rate=0.3,
    epochs=50,
    batch_size=32,
    verbose=0
)
cnn_lstm.fit(X_train, y_train)
cnn_lstm_metrics = evaluator.evaluate_model(cnn_lstm, X_test, y_test, 'CNN-LSTM')
print(f"   CNN-LSTM Test Accuracy: {cnn_lstm_metrics['Accuracy']:.4f}")

# 5. GRU Classifier
print("\\n[5] Training GRU Classifier...")
gru = GRUClassifier(
    gru_units=[64, 32],
    dropout_rate=0.3,
    epochs=50,
    batch_size=32,
    verbose=0
)
gru.fit(X_train, y_train)
gru_metrics = evaluator.evaluate_model(gru, X_test, y_test, 'GRU')
print(f"   GRU Test Accuracy: {gru_metrics['Accuracy']:.4f}")

# 6. Attention LSTM
print("\\n[6] Training Attention LSTM...")
att_lstm = AttentionLSTMClassifier(
    lstm_units=[64, 64],
    attention_units=64,
    dropout_rate=0.3,
    epochs=50,
    batch_size=32,
    verbose=0
)
att_lstm.fit(X_train, y_train)
att_lstm_metrics = evaluator.evaluate_model(att_lstm, X_test, y_test, 'Attention-LSTM')
print(f"   Attention-LSTM Test Accuracy: {att_lstm_metrics['Accuracy']:.4f}")

# Compare all models
print("\\n" + "="*60)
print("DEEP LEARNING MODELS COMPARISON")
print("="*60)
comparison_df = evaluator.compare_models()
print(comparison_df.to_string(index=False))
'''
    return code


def demo_deep_stacked_ensemble():
    """
    Example: Using Deep Stacked Ensemble with CNN-LSTM meta-learner.
    
    Copy this code into a notebook cell:
    """
    code = '''
# ============================================================
# DEEP STACKED ENSEMBLE WITH CNN-LSTM META-LEARNER
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from vehms import (
    DataLoader, DataPreprocessor, ModelEvaluator,
    DeepStackedEnsemble, CNNClassifier, LSTMClassifier,
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

# Load and preprocess data
loader = DataLoader()
df = loader.load_dataset('dataset/augmented_data_with_environment.csv')

preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

# Define base classifiers (mix of ML and DL)
base_classifiers = [
    ('RF', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
    ('XGB', XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric='mlogloss')),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('SVM', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)),
]

print("\\n" + "="*60)
print("DEEP STACKED ENSEMBLE TRAINING")
print("="*60)

# Create and train Deep Stacked Ensemble with CNN-LSTM meta-learner
deep_stack = DeepStackedEnsemble(
    base_classifiers=base_classifiers,
    meta_learner_type='cnn_lstm',  # Options: 'cnn', 'lstm', 'bilstm', 'cnn_lstm', 'gru', 'attention_lstm', 'lr'
    cv=5,
    use_probabilities=True,
    passthrough=False,
    verbose=1
)

deep_stack.fit(X_train, y_train)

# Display architecture
deep_stack.display_architecture()

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(deep_stack, X_test, y_test, 'Deep-Stack-CNN-LSTM')

print("\\n" + "="*60)
print("DEEP STACKED ENSEMBLE RESULTS")
print("="*60)
print(f"\\nTest Accuracy: {metrics['Accuracy']:.4f}")
print(f"Test Precision: {metrics['Precision']:.4f}")
print(f"Test AUC: {metrics['AUC']:.4f}")
print(f"Test RMSE: {metrics['RMSE']:.4f}")

# Compare base classifier accuracies
print("\\n[INFO] Base Classifier Accuracies:")
base_accs = deep_stack.get_base_accuracies(X_test, y_test)
for name, acc in base_accs.items():
    print(f"   {name}: {acc:.4f}")
'''
    return code


def demo_dynamic_deep_stacked_ensemble():
    """
    Example: Using Dynamic Deep Stacked Ensemble with automatic classifier selection.
    
    Copy this code into a notebook cell:
    """
    code = '''
# ============================================================
# DYNAMIC DEEP STACKED ENSEMBLE
# ============================================================
# Automatically selects optimal base classifiers

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from vehms import (
    DataLoader, DataPreprocessor, ModelEvaluator,
    DynamicDeepStackedEnsemble, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

# Load and preprocess data
loader = DataLoader()
df = loader.load_dataset('dataset/augmented_data_with_environment.csv')

preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

print("\\n" + "="*60)
print("DYNAMIC DEEP STACKED ENSEMBLE")
print("="*60)

# Create Dynamic Deep Stacked Ensemble
# It will automatically select the best classifiers
dynamic_stack = DynamicDeepStackedEnsemble(
    selection_method='combined',  # Options: 'performance', 'diversity', 'combined'
    top_k=5,                      # Number of base classifiers to select
    performance_weight=0.6,       # Weight for performance vs diversity
    meta_learner_type='cnn_lstm', # Deep learning meta-learner
    cv=5,
    verbose=1
)

dynamic_stack.fit(X_train, y_train)

# Display selection summary
dynamic_stack.display_selection_summary()

# Display architecture
dynamic_stack.display_architecture()

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(dynamic_stack, X_test, y_test, 'Dynamic-Deep-Stack')

print("\\n" + "="*60)
print("DYNAMIC DEEP STACKED ENSEMBLE RESULTS")
print("="*60)
print(f"\\nTest Accuracy: {metrics['Accuracy']:.4f}")
print(f"Test Precision: {metrics['Precision']:.4f}")
print(f"Test AUC: {metrics['AUC']:.4f}")
print(f"Test RMSE: {metrics['RMSE']:.4f}")
'''
    return code


def demo_compare_all_approaches():
    """
    Example: Comprehensive comparison of all approaches.
    
    Copy this code into a notebook cell:
    """
    code = '''
# ============================================================
# COMPREHENSIVE COMPARISON: ML vs DL vs Hybrid Stacking
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from vehms import (
    DataLoader, DataPreprocessor, ModelEvaluator,
    CNNClassifier, LSTMClassifier, CNNLSTMClassifier,
    DeepStackedEnsemble, DynamicDeepStackedEnsemble,
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

# Load and preprocess data
loader = DataLoader()
df = loader.load_dataset('dataset/augmented_data_with_environment.csv')

preprocessor = DataPreprocessor()
X, y = preprocessor.separate_features_target(df)
X_scaled = preprocessor.fit_transform_features(X)
y_encoded = preprocessor.encode_target(y)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_scaled, y_encoded)

evaluator = ModelEvaluator()
results = {}

print("\\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*70)

# 1. Traditional ML Stacking (Baseline)
print("\\n[1/5] Training Traditional ML Stacking...")
ml_stack = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric='mlogloss')),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)),
        ('ada', AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    cv=5
)
ml_stack.fit(X_train, y_train)
results['ML-Stack-LR'] = evaluator.evaluate_model(ml_stack, X_test, y_test, 'ML-Stack-LR')

# 2. Individual CNN
print("[2/5] Training CNN Classifier...")
cnn = CNNClassifier(epochs=50, verbose=0)
cnn.fit(X_train, y_train)
results['CNN'] = evaluator.evaluate_model(cnn, X_test, y_test, 'CNN')

# 3. Individual CNN-LSTM
print("[3/5] Training CNN-LSTM Classifier...")
cnn_lstm = CNNLSTMClassifier(epochs=50, verbose=0)
cnn_lstm.fit(X_train, y_train)
results['CNN-LSTM'] = evaluator.evaluate_model(cnn_lstm, X_test, y_test, 'CNN-LSTM')

# 4. Deep Stacked Ensemble (ML base + DL meta)
print("[4/5] Training Deep Stacked Ensemble...")
base_classifiers = [
    ('RF', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
    ('XGB', XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric='mlogloss')),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('SVM', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)),
]
deep_stack = DeepStackedEnsemble(
    base_classifiers=base_classifiers,
    meta_learner_type='cnn_lstm',
    cv=5,
    verbose=0
)
deep_stack.fit(X_train, y_train)
results['Deep-Stack-CNN-LSTM'] = evaluator.evaluate_model(deep_stack, X_test, y_test, 'Deep-Stack-CNN-LSTM')

# 5. Dynamic Deep Stacked Ensemble
print("[5/5] Training Dynamic Deep Stacked Ensemble...")
dynamic_stack = DynamicDeepStackedEnsemble(
    selection_method='combined',
    top_k=5,
    meta_learner_type='cnn_lstm',
    cv=5,
    verbose=0
)
dynamic_stack.fit(X_train, y_train)
results['Dynamic-Deep-Stack'] = evaluator.evaluate_model(dynamic_stack, X_test, y_test, 'Dynamic-Deep-Stack')

# Create comparison table
print("\\n" + "="*70)
print("FINAL RESULTS COMPARISON")
print("="*70)

comparison_df = evaluator.compare_models()
print("\\n" + comparison_df.to_string(index=False))

# Find best model
best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
print(f"\\n[BEST MODEL] {best_model['Model']}")
print(f"   Accuracy:  {best_model['Accuracy']:.4f}")
print(f"   Precision: {best_model['Precision']:.4f}")
print(f"   AUC:       {best_model['AUC']:.4f}")
print(f"   RMSE:      {best_model['RMSE']:.4f}")
'''
    return code


if __name__ == "__main__":
    print("="*60)
    print("VEHMS Deep Learning Classifiers - Demo Code")
    print("="*60)
    print("\nAvailable demos:")
    print("1. demo_individual_classifiers() - Individual DL models")
    print("2. demo_deep_stacked_ensemble() - Deep stacked ensemble")
    print("3. demo_dynamic_deep_stacked_ensemble() - Dynamic selection")
    print("4. demo_compare_all_approaches() - Full comparison")
    print("\nCall any function to get the code to paste into your notebook.")
