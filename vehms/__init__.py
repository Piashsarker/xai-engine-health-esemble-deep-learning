# ============================================================
# VEHMS - Vehicle Engine Health Monitoring System
# ============================================================
# Modular ML Pipeline for Predictive Maintenance
# ============================================================

from .data_loader import DataLoader
from .data_visualizer import DataVisualizer
from .data_cleaner import DataCleaner
from .data_preprocessor import DataPreprocessor
from .feature_extractor import FeatureExtractor
from .base_classifier_module import BaseClassifierModule
from .stacked_ensemble import StackedEnsemble
from .existing_research_stacked_ensemble import ExistingResearchStackedEnsemble
from .dynamic_classifier_selector import DynamicClassifierSelector
from .model_evaluator import ModelEvaluator
from .performance_visualizer import PerformanceVisualizer
from .xai_explainer import XAIExplainer
from .config import RANDOM_SEED, REQUIRED_COLUMNS, VALID_DECISIONS, TARGET_METRICS

# Deep Learning Classifiers
from .deep_learning_classifiers import (
    CNNClassifier,
    LSTMClassifier,
    CNNLSTMClassifier,
    GRUClassifier,
    AttentionLSTMClassifier
)

# Deep Stacked Ensembles
from .deep_stacked_ensemble import (
    DeepStackedEnsemble,
    DynamicDeepStackedEnsemble
)

# Hyperparameter Tuning
from .hyperparameter_tuner import (
    DeepLearningHyperparameterTuner,
    OptimizedHyperparameters,
    tune_all_models
)

# Deep Ensemble Voting
from .deep_ensemble_voting import (
    DeepEnsembleVoting,
    AdaptiveDeepEnsemble,
    create_optimized_ensemble
)

__all__ = [
    # Data Processing
    'DataLoader',
    'DataVisualizer', 
    'DataCleaner',
    'DataPreprocessor',
    'FeatureExtractor',
    
    # Traditional ML Classifiers
    'BaseClassifierModule',
    'StackedEnsemble',
    'ExistingResearchStackedEnsemble',
    'DynamicClassifierSelector',
    
    # Deep Learning Classifiers
    'CNNClassifier',
    'LSTMClassifier',
    'CNNLSTMClassifier',
    'GRUClassifier',
    'AttentionLSTMClassifier',
    
    # Deep Stacked Ensembles
    'DeepStackedEnsemble',
    'DynamicDeepStackedEnsemble',
    
    # Hyperparameter Tuning
    'DeepLearningHyperparameterTuner',
    'OptimizedHyperparameters',
    'tune_all_models',
    
    # Deep Ensemble Voting
    'DeepEnsembleVoting',
    'AdaptiveDeepEnsemble',
    'create_optimized_ensemble',
    
    # Evaluation & Visualization
    'ModelEvaluator',
    'PerformanceVisualizer',
    'XAIExplainer',
    
    # Configuration
    'RANDOM_SEED',
    'REQUIRED_COLUMNS',
    'VALID_DECISIONS',
    'TARGET_METRICS'
]

__version__ = '1.2.0'
