# ============================================================
# VEHMS Configuration
# ============================================================
# Central configuration for the VEHMS ML pipeline
# ============================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Required dataset columns
REQUIRED_COLUMNS = [
    'Crankshaft', 'Overheating', 'Lubricant', 'Misfires', 
    'Piston', 'Starter', 'Decision', 'Temperature', 
    'Humidity', 'Altitude'
]

# Valid decision categories
VALID_DECISIONS = ['Good', 'Minor', 'Moderate', 'Critical']

# Sensor feature columns
SENSOR_FEATURES = ['Crankshaft', 'Overheating', 'Lubricant', 'Misfires', 'Piston', 'Starter']

# Environmental feature columns
ENVIRONMENTAL_FEATURES = ['Temperature', 'Humidity', 'Altitude']

# Target performance metrics for Stacked Model 1
TARGET_METRICS = {
    'AUC': 0.9702,
    'RMSE': 0.3355,
    'Accuracy': 0.9470,
    'Precision': 0.9486
}

# Default test split ratio
DEFAULT_TEST_SIZE = 0.2

# Default cross-validation folds
DEFAULT_CV_FOLDS = 5
