# Requirements Document

## Introduction

The Vehicle Engine Health Monitoring System (VEHMS) is a predictive maintenance framework that uses stacked ensemble deep learning to analyze vehicle engine sensor data and environmental conditions. The system recognizes patterns and abnormalities that may suggest prospective engine problems, enabling proactive maintenance decisions with high accuracy. The framework processes 10 sensor and environmental features to classify engine health into four categories: Good, Minor, Moderate, and Critical.

## Glossary

- **VEHMS**: Vehicle Engine Health Monitoring System - the complete predictive maintenance framework
- **Data_Loader**: Component responsible for loading and validating the engine sensor dataset
- **Data_Visualizer**: Component that generates exploratory data analysis visualizations
- **Data_Cleaner**: Component that handles missing values, duplicates, and data quality issues
- **Data_Preprocessor**: Component that transforms raw data for model consumption
- **Feature_Extractor**: Component that identifies and selects relevant features for prediction
- **Base_Classifier**: Individual machine learning model (KNN, SVM, RF, AdaBoost, XGBoost) used in the ensemble
- **Stacked_Ensemble**: Meta-learning architecture combining multiple base classifiers with a meta-learner
- **Meta_Learner**: Final classifier that combines predictions from base classifiers
- **Model_Evaluator**: Component that calculates performance metrics (RMSE, MAE, AUC, Accuracy, Precision)
- **XAI_Explainer**: Explainable AI component providing interpretable predictions for end users using SHAP and LIME
- **SHAP**: SHapley Additive exPlanations - a game-theoretic approach for global and local model interpretability
- **LIME**: Local Interpretable Model-agnostic Explanations - a technique for explaining individual predictions by approximating the model locally
- **Engine_Health_Status**: Target classification categories (Good, Minor, Moderate, Critical)
- **Sensor_Features**: Input measurements including Crankshaft, Overheating, Lubricant, Misfires, Piston, Starter
- **Environmental_Features**: Contextual measurements including Temperature, Humidity, Altitude

## Requirements

### Requirement 1: Dataset Loading and Validation

**User Story:** As an AI/ML researcher, I want to load and validate the engine sensor dataset, so that I can ensure data integrity before analysis.

#### Acceptance Criteria

1. WHEN a CSV file path is provided, THE Data_Loader SHALL load the dataset into a pandas DataFrame
2. THE Data_Loader SHALL validate that all 10 required columns exist (Crankshaft, Overheating, Lubricant, Misfires, Piston, Starter, Decision, Temperature, Humidity, Altitude)
3. THE Data_Loader SHALL verify that numeric columns contain valid float values
4. THE Data_Loader SHALL verify that the Decision column contains only valid categories (Good, Minor, Moderate, Critical)
5. IF the dataset contains invalid or missing required columns, THEN THE Data_Loader SHALL raise a descriptive validation error
6. WHEN the dataset is loaded successfully, THE Data_Loader SHALL display dataset shape, column types, and basic statistics

### Requirement 2: Exploratory Data Analysis and Visualization

**User Story:** As an AI/ML researcher, I want to visualize the dataset distributions and relationships, so that I can understand data patterns before modeling.

#### Acceptance Criteria

1. WHEN the dataset is loaded, THE Data_Visualizer SHALL generate distribution plots for all numeric features
2. THE Data_Visualizer SHALL generate a correlation heatmap showing relationships between all numeric features
3. THE Data_Visualizer SHALL generate class distribution plots for the Decision target variable
4. THE Data_Visualizer SHALL generate box plots to identify outliers in sensor readings
5. THE Data_Visualizer SHALL generate pairwise scatter plots for feature relationships
6. WHEN visualizations are generated, THE Data_Visualizer SHALL display them inline in the Jupyter notebook

### Requirement 3: Data Cleaning

**User Story:** As an AI/ML researcher, I want to clean the dataset by handling missing values and anomalies, so that the data quality is suitable for model training.

#### Acceptance Criteria

1. WHEN the dataset is analyzed, THE Data_Cleaner SHALL identify and report the count of missing values per column
2. WHEN the dataset is analyzed, THE Data_Cleaner SHALL identify and report duplicate rows
3. IF missing values are detected in numeric columns, THEN THE Data_Cleaner SHALL impute them using appropriate statistical methods
4. IF duplicate rows are detected, THEN THE Data_Cleaner SHALL remove them and report the count removed
5. THE Data_Cleaner SHALL identify and handle outliers using statistical methods (IQR or z-score)
6. WHEN cleaning is complete, THE Data_Cleaner SHALL display a summary of all cleaning operations performed

### Requirement 4: Data Preprocessing and Transformation

**User Story:** As an AI/ML researcher, I want to preprocess and transform the data, so that it is suitable for machine learning model training.

#### Acceptance Criteria

1. THE Data_Preprocessor SHALL separate features (X) from the target variable (y)
2. THE Data_Preprocessor SHALL apply StandardScaler normalization to all numeric feature columns
3. THE Data_Preprocessor SHALL apply LabelEncoder to transform the Decision categorical target into numeric labels
4. THE Data_Preprocessor SHALL split the dataset into training (80%) and testing (20%) sets with stratified sampling
5. THE Data_Preprocessor SHALL set a random state for reproducibility of the train-test split
6. WHEN preprocessing is complete, THE Data_Preprocessor SHALL display the shapes of training and testing sets

### Requirement 5: Feature Extraction and Selection

**User Story:** As an AI/ML researcher, I want to extract and analyze feature importance, so that I can understand which features contribute most to predictions.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL calculate feature importance scores using tree-based methods
2. THE Feature_Extractor SHALL rank features by their importance to the prediction task
3. THE Feature_Extractor SHALL generate a feature importance visualization (bar chart)
4. THE Feature_Extractor SHALL identify the top contributing features for engine health prediction
5. WHEN feature extraction is complete, THE Feature_Extractor SHALL display the ranked feature importance table

### Requirement 6: Individual Base Classifier Training and Evaluation

**User Story:** As an AI/ML researcher, I want to train and evaluate individual base classifiers, so that I can compare their standalone performance before ensemble creation.

#### Acceptance Criteria

1. THE Base_Classifier module SHALL train a K-Nearest Neighbors (KNN) classifier on the training data
2. THE Base_Classifier module SHALL train a Support Vector Machine (SVM) classifier on the training data
3. THE Base_Classifier module SHALL train a Random Forest (RF) classifier on the training data
4. THE Base_Classifier module SHALL train an AdaBoost (Ada) classifier on the training data
5. THE Base_Classifier module SHALL train an XGBoost (XGB) classifier on the training data
6. WHEN each classifier is trained, THE Model_Evaluator SHALL calculate training and testing accuracy
7. WHEN each classifier is trained, THE Model_Evaluator SHALL generate predictions on the test set
8. THE Model_Evaluator SHALL display a comparison table of all base classifier accuracies

### Requirement 7: Stacked Ensemble Model Construction

**User Story:** As an AI/ML researcher, I want to construct stacked ensemble models with different configurations, so that I can achieve higher prediction accuracy through model combination.

#### Acceptance Criteria

1. THE Stacked_Ensemble SHALL import and configure the sklearn StackingClassifier library
2. THE Stacked_Ensemble SHALL create at least 3 different stacking configurations using combinations of base classifiers
3. WHEN creating Stacked Model 1, THE Stacked_Ensemble SHALL combine KNN, SVM, RF, AdaBoost, and XGBoost as base estimators
4. THE Stacked_Ensemble SHALL define a Meta_Learner (Logistic Regression or similar) for final predictions
5. THE Stacked_Ensemble SHALL use cross-validation for training base classifiers to prevent data leakage
6. WHEN stacking configurations are defined, THE Stacked_Ensemble SHALL display the architecture of each stacked model

### Requirement 8: Model Evaluation with Cross-Validation

**User Story:** As an AI/ML researcher, I want to evaluate models using cross-validation, so that I can obtain robust performance estimates.

#### Acceptance Criteria

1. THE Model_Evaluator SHALL perform k-fold cross-validation (k=5 or k=10) on all models
2. THE Model_Evaluator SHALL calculate cross-validation scores for accuracy, precision, and AUC
3. THE Model_Evaluator SHALL compute mean and standard deviation of cross-validation scores
4. THE Model_Evaluator SHALL compare cross-validation results across all base classifiers and stacked ensembles
5. WHEN cross-validation is complete, THE Model_Evaluator SHALL display a summary table of all cross-validation results

### Requirement 9: Comprehensive Model Performance Metrics

**User Story:** As an AI/ML researcher, I want to calculate comprehensive performance metrics, so that I can thoroughly evaluate model effectiveness.

#### Acceptance Criteria

1. THE Model_Evaluator SHALL calculate Root Mean Square Error (RMSE) for all models
2. THE Model_Evaluator SHALL calculate Root Mean Square Deviation (RMSD) for all models
3. THE Model_Evaluator SHALL calculate Mean Absolute Error (MAE) for all models
4. THE Model_Evaluator SHALL calculate Accuracy score for all models
5. THE Model_Evaluator SHALL calculate Precision score (weighted average for multi-class) for all models
6. THE Model_Evaluator SHALL calculate Area Under the Curve (AUC) using one-vs-rest strategy for multi-class
7. THE Model_Evaluator SHALL generate confusion matrices for all models
8. WHEN Stacked Model 1 is evaluated, THE Model_Evaluator SHALL verify performance targets (AUC ~0.9702, RMSE ~0.3355, Accuracy ~0.9470, Precision ~0.9486)

### Requirement 10: Model Performance Visualization and Comparison

**User Story:** As an AI/ML researcher, I want to visualize and compare model performance, so that I can identify the best performing model configuration.

#### Acceptance Criteria

1. THE Data_Visualizer SHALL generate bar charts comparing accuracy across all models
2. THE Data_Visualizer SHALL generate bar charts comparing RMSE across all models
3. THE Data_Visualizer SHALL generate bar charts comparing AUC across all models
4. THE Data_Visualizer SHALL generate ROC curves for all models on the same plot
5. THE Data_Visualizer SHALL generate confusion matrix heatmaps for each model
6. THE Data_Visualizer SHALL create a comprehensive comparison dashboard showing all metrics
7. WHEN visualizations are generated, THE Data_Visualizer SHALL highlight the best performing model

### Requirement 11: Final Model Prediction and Results

**User Story:** As an AI/ML researcher, I want to generate final predictions using the best model, so that I can demonstrate the system's predictive capability.

#### Acceptance Criteria

1. THE VEHMS SHALL select the best performing stacked ensemble model based on evaluation metrics
2. THE VEHMS SHALL generate final predictions on the test dataset using the selected model
3. THE VEHMS SHALL display a classification report with precision, recall, and F1-score per class
4. THE VEHMS SHALL display sample predictions with actual vs predicted labels
5. THE VEHMS SHALL calculate and display the overall prediction confidence scores
6. WHEN final predictions are complete, THE VEHMS SHALL summarize the model's performance against target metrics

### Requirement 12: Explainable AI (XAI) for End User Interpretability

**User Story:** As an AI/ML researcher, I want to provide explainable AI insights using both SHAP and LIME, so that end users can understand why specific predictions are made through multiple interpretability approaches.

#### Acceptance Criteria

##### SHAP (SHapley Additive exPlanations)

1. THE XAI_Explainer SHALL implement SHAP for global and local model interpretability
2. THE XAI_Explainer SHALL generate SHAP summary plots showing feature importance across all predictions
3. THE XAI_Explainer SHALL generate SHAP force plots for individual prediction explanations
4. THE XAI_Explainer SHALL generate SHAP waterfall plots to show cumulative feature contributions
5. THE XAI_Explainer SHALL generate SHAP dependence plots for the most important features
6. THE XAI_Explainer SHALL generate SHAP bar plots showing mean absolute SHAP values per feature

##### LIME (Local Interpretable Model-agnostic Explanations)

7. THE XAI_Explainer SHALL implement LIME for local instance-level explanations
8. THE XAI_Explainer SHALL create a LimeTabularExplainer configured for the engine sensor dataset
9. THE XAI_Explainer SHALL generate LIME explanations for sample predictions showing feature contributions
10. THE XAI_Explainer SHALL visualize LIME explanations as horizontal bar charts with positive/negative contributions
11. THE XAI_Explainer SHALL display LIME prediction probabilities for each Engine_Health_Status class
12. THE XAI_Explainer SHALL compare LIME and SHAP explanations for the same predictions to validate consistency

##### Combined XAI Analysis

13. THE XAI_Explainer SHALL identify which features most strongly influence each Engine_Health_Status category using both methods
14. THE XAI_Explainer SHALL provide natural language explanations for sample predictions combining SHAP and LIME insights
15. WHEN a prediction is made, THE XAI_Explainer SHALL display the top contributing factors from both SHAP and LIME
16. THE XAI_Explainer SHALL generate a comparison visualization showing SHAP vs LIME feature importance rankings

### Requirement 13: Input Validation for New Predictions

**User Story:** As an AI/ML researcher, I want to validate new input data before prediction, so that the system handles invalid inputs gracefully.

#### Acceptance Criteria

1. WHEN new sensor data is provided for prediction, THE Data_Loader SHALL validate that all required features are present
2. WHEN new sensor data is provided, THE Data_Loader SHALL validate that numeric values are within expected ranges
3. IF input data contains missing values, THEN THE VEHMS SHALL reject the prediction and return a descriptive error
4. IF input data contains out-of-range values, THEN THE VEHMS SHALL flag the anomaly and warn the user
5. THE VEHMS SHALL provide a prediction function that accepts a dictionary or DataFrame of sensor readings
6. WHEN valid input is provided, THE VEHMS SHALL return the predicted Engine_Health_Status with confidence score

### Requirement 14: Jupyter Notebook Implementation

**User Story:** As an AI/ML researcher, I want the complete VEHMS implemented in a Jupyter notebook, so that I can follow the data science lifecycle interactively.

#### Acceptance Criteria

1. THE VEHMS notebook SHALL be organized into clearly labeled sections following the 13-step workflow
2. THE VEHMS notebook SHALL include markdown documentation explaining each step
3. THE VEHMS notebook SHALL include code comments explaining key implementation decisions
4. THE VEHMS notebook SHALL execute sequentially without errors when run from top to bottom
5. THE VEHMS notebook SHALL display all visualizations inline
6. THE VEHMS notebook SHALL include a summary section with final results and conclusions
7. THE VEHMS notebook SHALL be saved with all outputs visible for review
