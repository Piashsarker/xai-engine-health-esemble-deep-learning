# ============================================================
# Phase 6 & 7: Hyperparameter Tuning and Ensemble Voting
# ============================================================
# Copy this code into your Jupyter notebook to run experiments
# ============================================================

"""
PHASE 6: HYPERPARAMETER TUNING
PHASE 7: ENSEMBLE VOTING FOR BETTER ACCURACY

This script provides ready-to-use code for improving your deep learning
model performance through systematic hyperparameter tuning and advanced
ensemble voting strategies.

USAGE:
------
1. Copy the relevant sections into your Jupyter notebook
2. Run after your data preprocessing steps
3. Compare results with existing research benchmarks
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error

# Import VEHMS modules
from vehms import (
    CNNClassifier, LSTMClassifier, CNNLSTMClassifier,
    GRUClassifier, AttentionLSTMClassifier,
    DeepStackedEnsemble, DynamicDeepStackedEnsemble,
    DeepLearningHyperparameterTuner, OptimizedHyperparameters,
    DeepEnsembleVoting, AdaptiveDeepEnsemble, create_optimized_ensemble,
    RANDOM_SEED
)

# Set seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# PHASE 6: HYPERPARAMETER TUNING
# ============================================================

def run_phase6_hyperparameter_tuning(X_train, y_train, quick_mode=True):
    """
    Run hyperparameter tuning for all deep learning models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (scaled)
    y_train : array-like
        Training labels (encoded)
    quick_mode : bool
        If True, use fewer iterations for faster results
        
    Returns:
    --------
    dict : Tuned parameters for each model
    """
    print("\n" + "="*70)
    print("PHASE 6: HYPERPARAMETER TUNING")
    print("="*70)
    
    n_iter = 5 if quick_mode else 15
    cv = 3 if quick_mode else 5
    
    tuned_params = {}
    
    # Models to tune (focus on underperforming ones)
    models_to_tune = ['lstm', 'bilstm', 'gru', 'attention_lstm']
    
    for model_type in models_to_tune:
        print(f"\n{'#'*60}")
        print(f"# Tuning: {model_type.upper()}")
        print(f"{'#'*60}")
        
        tuner = DeepLearningHyperparameterTuner(
            classifier_type=model_type,
            cv=cv,
            verbose=1
        )
        
        # Custom parameter grid for better performance
        if model_type in ['lstm', 'bilstm']:
            param_grid = {
                'lstm_units': [[128, 64], [256, 128], [128, 64, 32]],
                'dropout_rate': [0.2, 0.25, 0.3],
                'recurrent_dropout': [0.1, 0.15, 0.2],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64],
                'epochs': [150]
            }
        elif model_type == 'gru':
            param_grid = {
                'gru_units': [[128, 64], [256, 128], [128, 64, 32]],
                'dropout_rate': [0.2, 0.25, 0.3],
                'recurrent_dropout': [0.1, 0.15, 0.2],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64],
                'epochs': [150]
            }
        elif model_type == 'attention_lstm':
            param_grid = {
                'lstm_units': [[128, 64], [256, 128], [128, 128]],
                'attention_units': [32, 64, 128],
                'dropout_rate': [0.2, 0.25, 0.3],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64],
                'epochs': [150]
            }
        else:
            param_grid = None
        
        result = tuner.random_search(X_train, y_train, 
                                     param_distributions=param_grid,
                                     n_iter=n_iter)
        
        tuned_params[model_type] = {
            'params': result['best_params'],
            'cv_score': result['best_score'],
            'estimator': tuner.best_estimator_
        }
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 6 SUMMARY: TUNED HYPERPARAMETERS")
    print("="*70)
    
    for model, data in tuned_params.items():
        print(f"\n{model.upper()}:")
        print(f"  CV Score: {data['cv_score']:.4f}")
        print(f"  Best Params: {data['params']}")
    
    return tuned_params


# ============================================================
# PHASE 7: ENSEMBLE VOTING
# ============================================================

def run_phase7_ensemble_voting(X_train, y_train, X_test, y_test, 
                               tuned_params=None, verbose=1):
    """
    Run ensemble voting experiments with multiple strategies.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (scaled)
    y_train : array-like
        Training labels (encoded)
    X_test : array-like
        Test features (scaled)
    y_test : array-like
        Test labels (encoded)
    tuned_params : dict, optional
        Tuned hyperparameters from Phase 6
    verbose : int
        Verbosity level
        
    Returns:
    --------
    dict : Results for each ensemble configuration
    """
    print("\n" + "="*70)
    print("PHASE 7: ENSEMBLE VOTING")
    print("="*70)
    
    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, 
        stratify=y_train, random_state=RANDOM_SEED
    )
    
    results = {}
    
    # --------------------------------------------------------
    # Strategy 1: Best Performers Ensemble (CNN + CNN-LSTM)
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Strategy 1: Best Performers Ensemble")
    print("-"*60)
    
    estimators_best = [
        ('CNN', CNNClassifier(
            filters=[128, 256, 512],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            batch_size=32,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('CNN-LSTM', CNNLSTMClassifier(
            cnn_filters=[128, 256],
            lstm_units=[64],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            batch_size=32,
            random_state=RANDOM_SEED,
            verbose=0
        ))
    ]
    
    ensemble_best = DeepEnsembleVoting(
        estimators=estimators_best,
        voting='weighted_soft',
        verbose=verbose
    )
    ensemble_best.fit(X_tr, y_tr, X_val, y_val)
    
    y_pred = ensemble_best.predict(X_test)
    y_proba = ensemble_best.predict_proba(X_test)
    
    results['Best_Performers'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"Accuracy: {results['Best_Performers']['Accuracy']:.4f}")
    
    # --------------------------------------------------------
    # Strategy 2: Diverse Models Ensemble
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Strategy 2: Diverse Models Ensemble")
    print("-"*60)
    
    # Use tuned params if available
    lstm_params = tuned_params.get('lstm', {}).get('params', {}) if tuned_params else {}
    gru_params = tuned_params.get('gru', {}).get('params', {}) if tuned_params else {}
    
    estimators_diverse = [
        ('CNN', CNNClassifier(
            filters=[128, 256, 512],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('CNN-LSTM', CNNLSTMClassifier(
            cnn_filters=[128, 256],
            lstm_units=[64],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('LSTM', LSTMClassifier(
            lstm_units=lstm_params.get('lstm_units', [128, 64]),
            dropout_rate=lstm_params.get('dropout_rate', 0.25),
            recurrent_dropout=lstm_params.get('recurrent_dropout', 0.15),
            learning_rate=lstm_params.get('learning_rate', 0.0005),
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('GRU', GRUClassifier(
            gru_units=gru_params.get('gru_units', [128, 64]),
            dropout_rate=gru_params.get('dropout_rate', 0.25),
            recurrent_dropout=gru_params.get('recurrent_dropout', 0.15),
            learning_rate=gru_params.get('learning_rate', 0.0005),
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        ))
    ]
    
    ensemble_diverse = DeepEnsembleVoting(
        estimators=estimators_diverse,
        voting='weighted_soft',
        verbose=verbose
    )
    ensemble_diverse.fit(X_tr, y_tr, X_val, y_val)
    
    y_pred = ensemble_diverse.predict(X_test)
    y_proba = ensemble_diverse.predict_proba(X_test)
    
    results['Diverse_Models'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"Accuracy: {results['Diverse_Models']['Accuracy']:.4f}")
    
    # --------------------------------------------------------
    # Strategy 3: Adaptive Ensemble (Auto-select best strategy)
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Strategy 3: Adaptive Ensemble")
    print("-"*60)
    
    estimators_adaptive = [
        ('CNN', CNNClassifier(
            filters=[128, 256, 512],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('CNN-LSTM', CNNLSTMClassifier(
            cnn_filters=[128, 256],
            lstm_units=[64],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('BiLSTM', LSTMClassifier(
            lstm_units=[64, 32],
            bidirectional=True,
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        ))
    ]
    
    adaptive_ensemble = AdaptiveDeepEnsemble(
        estimators=estimators_adaptive,
        strategies=['hard', 'soft', 'weighted_soft', 'weighted_hard', 'stacked'],
        verbose=verbose
    )
    adaptive_ensemble.fit(X_tr, y_tr, X_val, y_val)
    
    y_pred = adaptive_ensemble.predict(X_test)
    y_proba = adaptive_ensemble.predict_proba(X_test)
    
    results['Adaptive'] = calculate_metrics(y_test, y_pred, y_proba)
    results['Adaptive']['best_strategy'] = adaptive_ensemble.best_strategy_
    print(f"Accuracy: {results['Adaptive']['Accuracy']:.4f}")
    print(f"Best Strategy: {adaptive_ensemble.best_strategy_}")
    
    # --------------------------------------------------------
    # Strategy 4: Stacked Ensemble with Deep Meta-Learner
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Strategy 4: Deep Stacked Ensemble")
    print("-"*60)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    
    base_classifiers = [
        ('RF', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
        ('XGB', XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, 
                              use_label_encoder=False, eval_metric='mlogloss')),
        ('GB', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED))
    ]
    
    deep_stack = DeepStackedEnsemble(
        base_classifiers=base_classifiers,
        meta_learner_type='cnn_lstm',
        cv=5,
        use_probabilities=True,
        passthrough=True,
        verbose=verbose
    )
    deep_stack.fit(X_train, y_train)
    
    y_pred = deep_stack.predict(X_test)
    y_proba = deep_stack.predict_proba(X_test)
    
    results['Deep_Stacked'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"Accuracy: {results['Deep_Stacked']['Accuracy']:.4f}")
    
    # --------------------------------------------------------
    # Strategy 5: Full Deep Learning Ensemble
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("Strategy 5: Full Deep Learning Ensemble")
    print("-"*60)
    
    estimators_full = [
        ('CNN', CNNClassifier(
            filters=[128, 256, 512],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('CNN-LSTM', CNNLSTMClassifier(
            cnn_filters=[128, 256],
            lstm_units=[64],
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('LSTM', LSTMClassifier(
            lstm_units=[128, 64],
            dropout_rate=0.25,
            recurrent_dropout=0.15,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('BiLSTM', LSTMClassifier(
            lstm_units=[64, 32],
            bidirectional=True,
            dropout_rate=0.25,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        )),
        ('GRU', GRUClassifier(
            gru_units=[128, 64],
            dropout_rate=0.25,
            recurrent_dropout=0.15,
            learning_rate=0.0005,
            epochs=150,
            random_state=RANDOM_SEED,
            verbose=0
        ))
    ]
    
    full_ensemble = DeepEnsembleVoting(
        estimators=estimators_full,
        voting='weighted_soft',
        verbose=verbose
    )
    full_ensemble.fit(X_tr, y_tr, X_val, y_val)
    
    y_pred = full_ensemble.predict(X_test)
    y_proba = full_ensemble.predict_proba(X_test)
    
    results['Full_DL_Ensemble'] = calculate_metrics(y_test, y_pred, y_proba)
    print(f"Accuracy: {results['Full_DL_Ensemble']['Accuracy']:.4f}")
    
    # Print comparison
    comparison = full_ensemble.compare_with_individuals(X_test, y_test)
    print("\n" + "-"*60)
    print("Individual vs Ensemble Comparison:")
    print("-"*60)
    print(comparison.to_string(index=False))
    
    return results


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'RMSD': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': np.mean(np.abs(y_true - y_pred))
    }
    
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    except:
        metrics['AUC'] = None
    
    return metrics


def print_final_results(results, existing_research_metrics=None):
    """Print final comparison of all results."""
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    # Create DataFrame
    df = pd.DataFrame(results).T
    df = df[['RMSE', 'RMSD', 'MAE', 'Accuracy', 'Precision', 'AUC']]
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    print("\n" + df.to_string())
    
    # Compare with existing research
    if existing_research_metrics:
        print("\n" + "-"*80)
        print("COMPARISON WITH EXISTING RESEARCH")
        print("-"*80)
        
        best_model = df.index[0]
        best_acc = df.loc[best_model, 'Accuracy']
        
        for name, metrics in existing_research_metrics.items():
            print(f"\n{name}:")
            print(f"  Research Accuracy: {metrics.get('Accuracy', 'N/A')}")
            print(f"  Our Best ({best_model}): {best_acc:.4f}")
            
            if 'Accuracy' in metrics:
                improvement = (best_acc - metrics['Accuracy']) * 100
                print(f"  Improvement: {improvement:+.2f}%")
    
    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    """
    Example usage - copy this into your Jupyter notebook after data preprocessing.
    
    Prerequisites:
    - X_train_scaled, X_test_scaled: Scaled feature arrays
    - y_train_encoded, y_test_encoded: Encoded label arrays
    """
    
    print("""
    ============================================================
    PHASE 6 & 7: HYPERPARAMETER TUNING AND ENSEMBLE VOTING
    ============================================================
    
    To run these experiments, add the following code to your notebook
    after your data preprocessing steps:
    
    # Phase 6: Hyperparameter Tuning (optional but recommended)
    tuned_params = run_phase6_hyperparameter_tuning(
        X_train_scaled, y_train_encoded, 
        quick_mode=True  # Set False for thorough tuning
    )
    
    # Phase 7: Ensemble Voting
    results = run_phase7_ensemble_voting(
        X_train_scaled, y_train_encoded,
        X_test_scaled, y_test_encoded,
        tuned_params=tuned_params,
        verbose=1
    )
    
    # Print final comparison
    existing_research = {
        'Stacked Model 1': {'Accuracy': 0.9470, 'AUC': 0.9702},
        'Stacked Model 2': {'Accuracy': 0.9380, 'AUC': 0.9650}
    }
    
    final_df = print_final_results(results, existing_research)
    ============================================================
    """)
