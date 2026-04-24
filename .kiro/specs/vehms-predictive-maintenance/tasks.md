# Implementation Plan: Vehicle Engine Health Monitoring System (VEHMS)

## Overview

This implementation plan covers the development of a predictive maintenance framework using stacked ensemble deep learning in a Jupyter notebook. The system analyzes vehicle engine sensor data and environmental conditions to classify engine health into four categories: Good, Minor, Moderate, and Critical. Target performance metrics: AUC ~0.9702, RMSE ~0.3355, Accuracy ~0.9470, Precision ~0.9486.

## Tasks

- [x] 1. Set up Jupyter notebook structure and library imports
  - [x] 1.1 Create the main notebook file with proper naming convention
    - Create `VEHMS_Predictive_Maintenance.ipynb` in the project root
    - Add notebook metadata and kernel configuration for Python 3.9+
    - _Requirements: 14.1, 14.2_

  - [x] 1.2 Implement library imports and configuration section
    - Import pandas, numpy, scipy for data processing
    - Import matplotlib, seaborn, plotly for visualization
    - Import sklearn components (preprocessing, model_selection, metrics, ensemble, neighbors, svm)
    - Import xgboost, shap, lime for ML and XAI
    - Set display options and random seed (42) for reproducibility
    - _Requirements: 14.1, 14.3_

- [x] 2. Implement DataLoader component
  - [x] 2.1 Create DataLoader class with validation methods
    - Implement `load_dataset()` method to read CSV files
    - Implement `validate_columns()` to check all 10 required columns exist
    - Implement `validate_data_types()` to verify numeric and categorical values
    - Implement `display_summary()` to show dataset shape, types, and statistics
    - Define REQUIRED_COLUMNS and VALID_DECISIONS class constants
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 2.2 Load and validate the engine sensor dataset
    - Load `dataset/augmented_data_with_environment.csv`
    - Execute validation checks and display summary statistics
    - Verify dataset contains 10,000 samples with 10 features
    - _Requirements: 1.1, 1.6_

- [x] 3. Implement DataVisualizer component for EDA
  - [x] 3.1 Create DataVisualizer class with visualization methods
    - Implement `plot_distributions()` for histogram distributions
    - Implement `plot_correlation_heatmap()` for feature correlations
    - Implement `plot_class_distribution()` for target variable distribution
    - Implement `plot_boxplots()` for outlier detection
    - Implement `plot_pairplot()` for pairwise scatter plots
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.2 Generate exploratory data analysis visualizations
    - Generate distribution plots for all 9 numeric features
    - Generate correlation heatmap showing feature relationships
    - Generate class distribution bar chart for Decision column
    - Generate box plots for sensor readings
    - Generate pairwise scatter plots colored by Decision
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 4. Implement DataCleaner component
  - [x] 4.1 Create DataCleaner class with cleaning methods
    - Implement `check_missing_values()` to identify missing values per column
    - Implement `check_duplicates()` to identify duplicate rows
    - Implement `impute_missing()` with median strategy for numeric columns
    - Implement `remove_duplicates()` to remove duplicate rows
    - Implement `handle_outliers()` using IQR method
    - Implement `get_cleaning_report()` to summarize operations
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 4.2 Execute data cleaning pipeline
    - Analyze and report missing values
    - Detect and remove duplicate rows
    - Handle outliers in sensor readings
    - Display cleaning summary report
    - _Requirements: 3.1, 3.2, 3.4, 3.6_

- [x] 5. Checkpoint - Verify data loading and cleaning
  - Ensure all data loading and cleaning steps execute without errors
  - Verify dataset integrity after cleaning operations
  - Ask the user if questions arise

- [x] 6. Implement DataPreprocessor component
  - [x] 6.1 Create DataPreprocessor class with transformation methods
    - Implement `separate_features_target()` to split X and y
    - Implement `fit_transform_features()` using StandardScaler
    - Implement `transform_features()` for new data transformation
    - Implement `encode_target()` using LabelEncoder
    - Implement `decode_target()` to convert labels back to categories
    - Implement `train_test_split()` with stratified sampling (80/20 split)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 6.2 Execute preprocessing pipeline
    - Separate features (9 columns) from target (Decision)
    - Apply StandardScaler to all numeric features
    - Apply LabelEncoder to Decision column (Good, Minor, Moderate, Critical → 0, 1, 2, 3)
    - Split into training (8000 samples) and testing (2000 samples) sets
    - Display shapes of X_train, X_test, y_train, y_test
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 7. Implement FeatureExtractor component
  - [x] 7.1 Create FeatureExtractor class
    - Implement `calculate_importance()` using RandomForestClassifier
    - Implement `rank_features()` to sort by importance score
    - Implement `plot_importance()` to generate bar chart
    - Implement `get_top_features()` to return top n features
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 7.2 Execute feature extraction and analysis
    - Calculate feature importance scores on training data
    - Generate ranked feature importance table
    - Create feature importance visualization
    - Identify top contributing features for engine health prediction
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 8. Implement BaseClassifierModule
  - [x] 8.1 Create BaseClassifierModule class
    - Define classifier dictionary with KNN (k=5), SVM (RBF kernel), RF (n=100), AdaBoost (n=100), XGBoost (n=100)
    - Implement `train_classifier()` for individual classifier training
    - Implement `train_all()` to train all base classifiers
    - Implement `predict()` and `predict_proba()` methods
    - Implement `get_classifier()` to retrieve trained models
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 8.2 Train and evaluate all base classifiers
    - Train KNN classifier on training data
    - Train SVM classifier on training data
    - Train Random Forest classifier on training data
    - Train AdaBoost classifier on training data
    - Train XGBoost classifier on training data
    - Calculate training and testing accuracy for each
    - Display comparison table of base classifier accuracies
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

- [x] 9. Checkpoint - Verify base classifiers
  - Ensure all 5 base classifiers train successfully
  - Verify predictions can be generated on test set
  - Ask the user if questions arise

- [x] 10. Implement StackedEnsemble component
  - [x] 10.1 Create StackedEnsemble class with 3 configurations
    - Implement `create_stacked_model_1()`: KNN + SVM + RF + AdaBoost + XGBoost with LogisticRegression meta-learner
    - Implement `create_stacked_model_2()`: RF + XGBoost + AdaBoost with LogisticRegression meta-learner
    - Implement `create_stacked_model_3()`: KNN + SVM + RF with LogisticRegression meta-learner
    - Implement `train_stacked_model()` with 5-fold cross-validation
    - Implement `display_architecture()` to show model structure
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 10.2 Train all stacked ensemble models
    - Create and train Stacked Model 1 (all 5 base classifiers)
    - Create and train Stacked Model 2 (RF + XGBoost + AdaBoost)
    - Create and train Stacked Model 3 (KNN + SVM + RF)
    - Display architecture diagrams for each stacked model
    - _Requirements: 7.2, 7.3, 7.5, 7.6_

- [x] 11. Implement ModelEvaluator component
  - [ ] 11.1 Create ModelEvaluator class with metric calculations
    - Implement `calculate_rmse()` using mean_squared_error
    - Implement `calculate_rmsd()` for root mean square deviation
    - Implement `calculate_mae()` using mean_absolute_error
    - Implement `calculate_accuracy()` using accuracy_score
    - Implement `calculate_precision()` using weighted precision_score
    - Implement `calculate_auc()` using one-vs-rest roc_auc_score
    - Implement `generate_confusion_matrix()` using confusion_matrix
    - Implement `cross_validate()` for k-fold cross-validation
    - Implement `evaluate_model()` to calculate all metrics for a model
    - Implement `compare_models()` to generate comparison DataFrame
    - Implement `verify_targets()` to check against target metrics
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ] 11.2 Perform cross-validation evaluation
    - Execute 5-fold cross-validation on all base classifiers
    - Execute 5-fold cross-validation on all stacked ensembles
    - Calculate mean and standard deviation of CV scores
    - Display CV results summary table
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 11.3 Calculate comprehensive metrics for all models
    - Calculate RMSE, RMSD, MAE for all models
    - Calculate Accuracy, Precision, AUC for all models
    - Generate confusion matrices for all models
    - Verify Stacked Model 1 meets targets (AUC ~0.9702, RMSE ~0.3355, Accuracy ~0.9470, Precision ~0.9486)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [x] 12. Implement performance visualization methods
  - [x] 12.1 Add visualization methods to DataVisualizer
    - Implement `plot_model_comparison()` for metric bar charts
    - Implement `plot_roc_curves()` for all models on same plot
    - Implement `plot_confusion_matrices()` for heatmap visualizations
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 12.2 Generate performance comparison visualizations
    - Generate accuracy comparison bar chart across all models
    - Generate RMSE comparison bar chart across all models
    - Generate AUC comparison bar chart across all models
    - Generate ROC curves for all models
    - Generate confusion matrix heatmaps for each model
    - Create comprehensive comparison dashboard
    - Highlight best performing model (Stacked Model 1)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 13. Checkpoint - Verify model training and evaluation
  - Ensure all models are trained and evaluated
  - Verify performance metrics are calculated correctly
  - Confirm Stacked Model 1 meets target performance
  - Ask the user if questions arise

- [x] 14. Implement final model selection and predictions
  - [x] 14.1 Select best model and generate final predictions
    - Select Stacked Model 1 as best performing model based on metrics
    - Generate final predictions on test dataset
    - Display classification report with precision, recall, F1-score per class
    - Display sample predictions with actual vs predicted labels
    - Calculate and display prediction confidence scores
    - Summarize performance against target metrics
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [x] 15. Implement XAI with SHAP
  - [x] 15.1 Create XAIExplainer class with SHAP methods
    - Implement `initialize_shap()` using TreeExplainer for ensemble model
    - Implement `calculate_shap_values()` for given samples
    - Implement `plot_shap_summary()` for beeswarm plot
    - Implement `plot_shap_bar()` for mean absolute SHAP values
    - Implement `plot_shap_force()` for individual prediction explanations
    - Implement `plot_shap_waterfall()` for cumulative contributions
    - Implement `plot_shap_dependence()` for feature dependence plots
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [x] 15.2 Generate SHAP analysis visualizations
    - Initialize SHAP explainer for best stacked model
    - Calculate SHAP values for test set samples
    - Generate SHAP summary plot showing feature importance
    - Generate SHAP bar plot with mean absolute values
    - Generate SHAP force plots for sample predictions
    - Generate SHAP waterfall plots for cumulative contributions
    - Generate SHAP dependence plots for top features
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [x] 16. Implement XAI with LIME
  - [x] 16.1 Add LIME methods to XAIExplainer class
    - Implement `initialize_lime()` using LimeTabularExplainer
    - Implement `explain_instance_lime()` for single instance explanations
    - Implement `plot_lime_explanation()` for horizontal bar chart visualization
    - Implement `get_lime_feature_weights()` to extract feature weights
    - Implement `get_lime_prediction_proba()` for prediction probabilities
    - _Requirements: 12.7, 12.8, 12.9, 12.10, 12.11_

  - [x] 16.2 Generate LIME analysis visualizations
    - Initialize LIME explainer with feature names and class names
    - Generate LIME explanations for sample predictions
    - Visualize LIME explanations as horizontal bar charts
    - Display LIME prediction probabilities for each Engine_Health_Status class
    - _Requirements: 12.7, 12.8, 12.9, 12.10, 12.11_

- [x] 17. Implement combined XAI analysis
  - [x] 17.1 Add combined analysis methods to XAIExplainer
    - Implement `compare_shap_lime()` to compare feature importance rankings
    - Implement `generate_natural_language_explanation()` combining SHAP and LIME insights
    - Implement `plot_comparison_visualization()` for side-by-side SHAP vs LIME
    - Implement `analyze_class_influences()` to identify features influencing each category
    - _Requirements: 12.12, 12.13, 12.14, 12.15, 12.16_

  - [x] 17.2 Generate combined XAI analysis
    - Compare SHAP and LIME explanations for same predictions
    - Generate comparison visualization showing SHAP vs LIME rankings
    - Identify features influencing each Engine_Health_Status category
    - Generate natural language explanations for sample predictions
    - Display top contributing factors from both SHAP and LIME
    - _Requirements: 12.12, 12.13, 12.14, 12.15, 12.16_

- [x] 18. Checkpoint - Verify XAI implementation
  - Ensure SHAP and LIME explainers work correctly
  - Verify all XAI visualizations render properly
  - Confirm combined analysis provides meaningful insights
  - Ask the user if questions arise

- [x] 19. Implement input validation for new predictions
  - [ ] 19.1 Add prediction validation methods
    - Implement input validation to check all required features are present
    - Implement range validation for numeric values
    - Implement error handling for missing values
    - Implement warning system for out-of-range values
    - Create prediction function accepting dictionary or DataFrame input
    - Return predicted Engine_Health_Status with confidence score
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [ ] 19.2 Test prediction function with sample inputs
    - Test with valid sensor readings
    - Test with missing values (should reject with error)
    - Test with out-of-range values (should warn user)
    - Demonstrate prediction with confidence score output
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

- [ ] 20. Add notebook documentation and conclusions
  - [ ] 20.1 Add markdown documentation throughout notebook
    - Add section headers following the 13-step workflow structure
    - Add explanatory markdown cells for each major section
    - Add code comments explaining key implementation decisions
    - Document the stacked ensemble architecture
    - _Requirements: 14.2, 14.3_

  - [ ] 20.2 Create summary and conclusions section
    - Summarize final results and model performance
    - Compare achieved metrics against target metrics
    - Document key findings from XAI analysis
    - Provide recommendations for model deployment
    - _Requirements: 14.6_

- [ ] 21. Final checkpoint - Complete notebook verification
  - Execute notebook from top to bottom without errors
  - Verify all visualizations display inline
  - Confirm all outputs are visible for review
  - Verify notebook follows the 13-step workflow structure
  - Ask the user if questions arise

## Notes

- This implementation is a Jupyter notebook following the data science lifecycle
- All components are implemented as Python classes within the notebook
- The target performance metrics for Stacked Model 1 are: AUC ~0.9702, RMSE ~0.3355, Accuracy ~0.9470, Precision ~0.9486
- Random state is set to 42 throughout for reproducibility
- The dataset contains 10,000 samples with 9 features and 1 target variable
- XAI implementation uses both SHAP (global + local) and LIME (local) for comprehensive interpretability
- Checkpoints ensure incremental validation of the implementation
