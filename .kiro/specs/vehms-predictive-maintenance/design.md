# Technical Design Document

## Vehicle Engine Health Monitoring System (VEHMS)

### 1. Overview

This document describes the technical design for the Vehicle Engine Health Monitoring System (VEHMS), a predictive maintenance framework using stacked ensemble deep learning. The system analyzes vehicle engine sensor data and environmental conditions to classify engine health into four categories: Good, Minor, Moderate, and Critical.

### 2. System Architecture

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
│  │ Data_Loader  │    │Data_Cleaner  │    │Base_Classifier│                  │
│  │Data_Visualizer│   │Data_Preproc  │    │Stacked_Ensemble│                 │
│  └──────────────┘    │Feature_Extr  │    │Model_Evaluator│                  │
│                      └──────────────┘    └──────────────┘                   │
│                                                 │                           │
│                                                 ▼                           │
│                                          ┌──────────────┐                   │
│                                          │ XAI Layer    │                   │
│                                          │ SHAP + LIME  │                   │
│                                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Data Model

#### 3.1 Input Dataset Schema

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| Crankshaft | float64 | Crankshaft sensor reading | 4.0 - 21.0 |
| Overheating | float64 | Overheating indicator | 8.0 - 26.0 |
| Lubricant | float64 | Lubricant level/quality | 4.0 - 16.0 |
| Misfires | float64 | Engine misfire intensity | 2.5 - 11.0 |
| Piston | float64 | Piston sensor reading | 3.5 - 16.0 |
| Starter | float64 | Starter motor measurement | 250.0 - 450.0 |
| Decision | categorical | Target: Engine health status | Good, Minor, Moderate, Critical |
| Temperature | float64 | Environmental temperature (°C) | 15.0 - 35.0 |
| Humidity | float64 | Environmental humidity (%) | 30.0 - 90.0 |
| Altitude | float64 | Environmental altitude (m) | 0.0 - 3000.0 |

#### 3.2 Processed Data Structure

```python
# Feature matrix after preprocessing
X_train: np.ndarray  # Shape: (8000, 9) - 80% of 10,000 samples
X_test: np.ndarray   # Shape: (2000, 9) - 20% of 10,000 samples

# Target vector after label encoding
y_train: np.ndarray  # Shape: (8000,) - Labels: 0, 1, 2, 3
y_test: np.ndarray   # Shape: (2000,) - Labels: 0, 1, 2, 3

# Label mapping
label_mapping = {0: 'Critical', 1: 'Good', 2: 'Minor', 3: 'Moderate'}
```

### 4. Component Design

#### 4.1 Data_Loader Component

**Purpose:** Load and validate the engine sensor dataset

**Implementation:**
```python
class DataLoader:
    REQUIRED_COLUMNS = ['Crankshaft', 'Overheating', 'Lubricant', 'Misfires', 
                        'Piston', 'Starter', 'Decision', 'Temperature', 
                        'Humidity', 'Altitude']
    VALID_DECISIONS = ['Good', 'Minor', 'Moderate', 'Critical']
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load CSV and validate structure"""
        
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Verify all required columns exist"""
        
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """Verify numeric columns and categorical values"""
        
    def display_summary(self, df: pd.DataFrame) -> None:
        """Display dataset shape, types, and statistics"""
```

**Dependencies:** pandas, numpy

#### 4.2 Data_Visualizer Component

**Purpose:** Generate exploratory data analysis visualizations

**Implementation:**
```python
class DataVisualizer:
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        
    def plot_distributions(self, df: pd.DataFrame, columns: list) -> None:
        """Generate histogram distributions for numeric features"""
        
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Generate correlation matrix heatmap"""
        
    def plot_class_distribution(self, df: pd.DataFrame, target: str) -> None:
        """Generate bar chart for target class distribution"""
        
    def plot_boxplots(self, df: pd.DataFrame, columns: list) -> None:
        """Generate box plots for outlier detection"""
        
    def plot_pairplot(self, df: pd.DataFrame, hue: str) -> None:
        """Generate pairwise scatter plots"""
        
    def plot_model_comparison(self, metrics_df: pd.DataFrame) -> None:
        """Generate model performance comparison charts"""
        
    def plot_roc_curves(self, models: dict, X_test, y_test) -> None:
        """Generate ROC curves for all models"""
        
    def plot_confusion_matrices(self, models: dict, X_test, y_test) -> None:
        """Generate confusion matrix heatmaps"""
```

**Dependencies:** matplotlib, seaborn, plotly

#### 4.3 Data_Cleaner Component

**Purpose:** Handle missing values, duplicates, and outliers

**Implementation:**
```python
class DataCleaner:
    def __init__(self, outlier_method: str = 'iqr'):
        self.outlier_method = outlier_method
        self.cleaning_report = {}
        
    def check_missing_values(self, df: pd.DataFrame) -> pd.Series:
        """Identify and count missing values per column"""
        
    def check_duplicates(self, df: pd.DataFrame) -> int:
        """Identify duplicate rows"""
        
    def impute_missing(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Impute missing values using specified strategy"""
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        
    def handle_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Detect and handle outliers using IQR or z-score"""
        
    def get_cleaning_report(self) -> dict:
        """Return summary of all cleaning operations"""
```

**Dependencies:** pandas, numpy, scipy

#### 4.4 Data_Preprocessor Component

**Purpose:** Transform data for machine learning model training

**Implementation:**
```python
class DataPreprocessor:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def separate_features_target(self, df: pd.DataFrame, target: str) -> tuple:
        """Separate features (X) from target variable (y)"""
        
    def fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Apply StandardScaler to numeric features"""
        
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler"""
        
    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Apply LabelEncoder to categorical target"""
        
    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode numeric labels back to categories"""
        
    def train_test_split(self, X, y) -> tuple:
        """Split data with stratified sampling"""
```

**Dependencies:** sklearn.preprocessing, sklearn.model_selection

#### 4.5 Feature_Extractor Component

**Purpose:** Calculate and analyze feature importance

**Implementation:**
```python
class FeatureExtractor:
    def __init__(self):
        self.importance_scores = None
        self.feature_names = None
        
    def calculate_importance(self, X, y, method: str = 'random_forest') -> pd.DataFrame:
        """Calculate feature importance using tree-based methods"""
        
    def rank_features(self) -> pd.DataFrame:
        """Rank features by importance score"""
        
    def plot_importance(self) -> None:
        """Generate feature importance bar chart"""
        
    def get_top_features(self, n: int = 5) -> list:
        """Return top n most important features"""
```

**Dependencies:** sklearn.ensemble.RandomForestClassifier

#### 4.6 Base_Classifier Module

**Purpose:** Train and evaluate individual classifiers

**Implementation:**
```python
class BaseClassifierModule:
    def __init__(self):
        self.classifiers = {
            'LR': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'LDA': LinearDiscriminantAnalysis(),
            'GNB': GaussianNB(),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'DT': DecisionTreeClassifier(random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, 
                                      use_label_encoder=False, eval_metric='mlogloss')
        }
        self.trained_models = {}
        
    def train_classifier(self, name: str, X_train, y_train) -> None:
        """Train a specific classifier"""
        
    def train_all(self, X_train, y_train) -> None:
        """Train all base classifiers"""
        
    def predict(self, name: str, X) -> np.ndarray:
        """Generate predictions using specified classifier"""
        
    def predict_proba(self, name: str, X) -> np.ndarray:
        """Generate probability predictions"""
        
    def get_classifier(self, name: str):
        """Return trained classifier instance"""
```

**Available Classifiers:**

| Classifier | Abbreviation | Key Parameters |
|------------|--------------|----------------|
| Logistic Regression | LR | max_iter=1000 |
| K-Nearest Neighbors | KNN | n_neighbors=5 |
| Linear Discriminant Analysis | LDA | - |
| Gaussian Naive Bayes | GNB | - |
| Support Vector Machine | SVM | kernel='rbf', probability=True |
| Decision Tree | DT | - |
| Random Forest | RF | n_estimators=100 |
| AdaBoost | AdaBoost | n_estimators=100 |
| Gradient Boosting | GB | n_estimators=100 |
| XGBoost | XGBoost | n_estimators=100 |

**Dependencies:** sklearn.neighbors, sklearn.svm, sklearn.ensemble, sklearn.linear_model, sklearn.discriminant_analysis, sklearn.naive_bayes, sklearn.tree, xgboost

#### 4.7 Stacked_Ensemble Component

**Purpose:** Construct stacked ensemble models with different configurations

**Implementation:**
```python
class StackedEnsemble:
    def __init__(self, cv: int = 5):
        self.cv = cv
        self.stacked_models = {}
        
    def create_stacked_model_1(self) -> StackingClassifier:
        """Create Stacked Model 1: KNN + SVM + RF + AdaBoost + XGBoost"""
        estimators = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=42))
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=self.cv
        )
        
    def create_stacked_model_2(self) -> StackingClassifier:
        """Create Stacked Model 2: RF + XGBoost + AdaBoost"""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=self.cv
        )
        
    def create_stacked_model_3(self) -> StackingClassifier:
        """Create Stacked Model 3: KNN + SVM + RF"""
        estimators = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=self.cv
        )
        
    def train_stacked_model(self, name: str, model, X_train, y_train) -> None:
        """Train a stacked ensemble model"""
        
    def display_architecture(self, name: str) -> None:
        """Display the architecture of a stacked model"""
```

**Dependencies:** sklearn.ensemble.StackingClassifier, sklearn.linear_model.LogisticRegression

#### 4.8 Model_Evaluator Component

**Purpose:** Calculate comprehensive performance metrics

**Implementation:**
```python
class ModelEvaluator:
    def __init__(self):
        self.metrics_results = {}
        
    def calculate_rmse(self, y_true, y_pred) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
        
    def calculate_rmsd(self, y_true, y_pred) -> float:
        """Calculate Root Mean Square Deviation"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
    def calculate_mae(self, y_true, y_pred) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
        
    def calculate_accuracy(self, y_true, y_pred) -> float:
        """Calculate Accuracy score"""
        return accuracy_score(y_true, y_pred)
        
    def calculate_precision(self, y_true, y_pred) -> float:
        """Calculate weighted Precision score"""
        return precision_score(y_true, y_pred, average='weighted')
        
    def calculate_auc(self, y_true, y_proba, n_classes: int = 4) -> float:
        """Calculate AUC using one-vs-rest strategy"""
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        return roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted')
        
    def generate_confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        """Generate confusion matrix"""
        return confusion_matrix(y_true, y_pred)
        
    def cross_validate(self, model, X, y, cv: int = 5, scoring: list = None) -> dict:
        """Perform k-fold cross-validation"""
        
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """Calculate all metrics for a model"""
        
    def compare_models(self) -> pd.DataFrame:
        """Generate comparison table of all model metrics"""
        
    def verify_targets(self, model_name: str, metrics: dict) -> bool:
        """Verify if model meets target performance metrics"""
```

**Dependencies:** sklearn.metrics, numpy


#### 4.9 XAI_Explainer Component

**Purpose:** Provide explainable AI insights using SHAP and LIME

**Implementation:**
```python
class XAIExplainer:
    def __init__(self, model, X_train, feature_names: list):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        
    # ==================== SHAP Methods ====================
    
    def initialize_shap(self) -> None:
        """Initialize SHAP explainer for the model"""
        self.shap_explainer = shap.TreeExplainer(self.model)
        # For non-tree models, use KernelExplainer:
        # self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, 
        #                                            shap.sample(self.X_train, 100))
        
    def calculate_shap_values(self, X) -> np.ndarray:
        """Calculate SHAP values for given samples"""
        return self.shap_explainer.shap_values(X)
        
    def plot_shap_summary(self, X, shap_values) -> None:
        """Generate SHAP summary plot (beeswarm)"""
        shap.summary_plot(shap_values, X, feature_names=self.feature_names)
        
    def plot_shap_bar(self, shap_values) -> None:
        """Generate SHAP bar plot showing mean absolute values"""
        shap.summary_plot(shap_values, plot_type='bar', 
                          feature_names=self.feature_names)
        
    def plot_shap_force(self, X_instance, shap_values_instance, 
                        expected_value) -> None:
        """Generate SHAP force plot for individual prediction"""
        shap.force_plot(expected_value, shap_values_instance, X_instance,
                        feature_names=self.feature_names, matplotlib=True)
        
    def plot_shap_waterfall(self, shap_values_instance, expected_value) -> None:
        """Generate SHAP waterfall plot for cumulative contributions"""
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_instance,
            base_values=expected_value,
            feature_names=self.feature_names
        ))
        
    def plot_shap_dependence(self, X, shap_values, feature: str) -> None:
        """Generate SHAP dependence plot for a specific feature"""
        feature_idx = self.feature_names.index(feature)
        shap.dependence_plot(feature_idx, shap_values, X,
                             feature_names=self.feature_names)
    
    # ==================== LIME Methods ====================
    
    def initialize_lime(self, class_names: list) -> None:
        """Initialize LIME explainer for tabular data"""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
        
    def explain_instance_lime(self, X_instance, num_features: int = 9) -> lime.explanation.Explanation:
        """Generate LIME explanation for a single instance"""
        return self.lime_explainer.explain_instance(
            X_instance,
            self.model.predict_proba,
            num_features=num_features,
            top_labels=4
        )
        
    def plot_lime_explanation(self, explanation, label: int = None) -> None:
        """Visualize LIME explanation as horizontal bar chart"""
        if label is not None:
            explanation.as_pyplot_figure(label=label)
        else:
            explanation.show_in_notebook()
            
    def get_lime_feature_weights(self, explanation, label: int) -> list:
        """Extract feature weights from LIME explanation"""
        return explanation.as_list(label=label)
        
    def get_lime_prediction_proba(self, explanation) -> dict:
        """Get prediction probabilities from LIME explanation"""
        return dict(zip(explanation.class_names, 
                       explanation.predict_proba.tolist()))
    
    # ==================== Combined XAI Analysis ====================
    
    def compare_shap_lime(self, X_instance, shap_values_instance, 
                          lime_explanation, label: int) -> pd.DataFrame:
        """Compare SHAP and LIME feature importance for same prediction"""
        # Get SHAP importance
        shap_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Value': np.abs(shap_values_instance)
        }).sort_values('SHAP_Value', ascending=False)
        
        # Get LIME importance
        lime_weights = self.get_lime_feature_weights(lime_explanation, label)
        lime_df = pd.DataFrame(lime_weights, columns=['Feature_Condition', 'LIME_Weight'])
        
        return shap_importance, lime_df
        
    def generate_natural_language_explanation(self, X_instance, prediction, 
                                               shap_values, lime_explanation) -> str:
        """Generate human-readable explanation combining SHAP and LIME"""
        # Get top 3 SHAP features
        shap_top = np.argsort(np.abs(shap_values))[-3:][::-1]
        shap_features = [self.feature_names[i] for i in shap_top]
        
        # Get top 3 LIME features
        lime_weights = lime_explanation.as_list(label=prediction)[:3]
        
        explanation = f"""
        **Prediction: {prediction}**
        
        **SHAP Analysis:**
        The top contributing factors according to SHAP are:
        1. {shap_features[0]} (contribution: {shap_values[shap_top[0]]:.4f})
        2. {shap_features[1]} (contribution: {shap_values[shap_top[1]]:.4f})
        3. {shap_features[2]} (contribution: {shap_values[shap_top[2]]:.4f})
        
        **LIME Analysis:**
        The local model approximation indicates:
        1. {lime_weights[0][0]} (weight: {lime_weights[0][1]:.4f})
        2. {lime_weights[1][0]} (weight: {lime_weights[1][1]:.4f})
        3. {lime_weights[2][0]} (weight: {lime_weights[2][1]:.4f})
        """
        return explanation
        
    def plot_comparison_visualization(self, shap_importance: pd.DataFrame, 
                                       lime_importance: pd.DataFrame) -> None:
        """Generate side-by-side comparison of SHAP vs LIME rankings"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # SHAP bar chart
        axes[0].barh(shap_importance['Feature'], shap_importance['SHAP_Value'])
        axes[0].set_title('SHAP Feature Importance')
        axes[0].set_xlabel('Mean |SHAP Value|')
        
        # LIME bar chart
        axes[1].barh(range(len(lime_importance)), lime_importance['LIME_Weight'])
        axes[1].set_yticks(range(len(lime_importance)))
        axes[1].set_yticklabels(lime_importance['Feature_Condition'])
        axes[1].set_title('LIME Feature Weights')
        axes[1].set_xlabel('Weight')
        
        plt.tight_layout()
        plt.show()
        
    def analyze_class_influences(self, X, y, class_names: list) -> dict:
        """Identify features influencing each Engine_Health_Status category"""
        class_influences = {}
        
        for class_idx, class_name in enumerate(class_names):
            class_mask = y == class_idx
            X_class = X[class_mask]
            
            # Calculate SHAP values for this class
            shap_values = self.calculate_shap_values(X_class)
            
            # Get mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values[class_idx]), axis=0)
            
            class_influences[class_name] = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean_SHAP': mean_shap
            }).sort_values('Mean_SHAP', ascending=False)
            
        return class_influences
```

**Dependencies:** shap, lime, matplotlib, pandas, numpy

### 5. Stacked Ensemble Architecture

#### 5.1 Stacked Model 1 (Primary - Target: AUC ~0.9702)

```
                    ┌─────────────────────────────────────┐
                    │         Input Features (9)          │
                    │  Crankshaft, Overheating, Lubricant │
                    │  Misfires, Piston, Starter,         │
                    │  Temperature, Humidity, Altitude    │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │    KNN    │   │    SVM    │   │    RF     │
              │  k=5      │   │  RBF      │   │ n=100     │
              └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                    │                │                │
              ┌─────▼─────┐   ┌─────▼─────┐          │
              │  AdaBoost │   │  XGBoost  │          │
              │  n=100    │   │  n=100    │          │
              └─────┬─────┘   └─────┬─────┘          │
                    │                │                │
                    └────────────────┼────────────────┘
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

#### 5.2 Stacked Model 2

```
              ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
              │     RF      │   │   XGBoost   │   │  AdaBoost   │
              │   n=100     │   │   n=100     │   │   n=100     │
              └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                     │                 │                 │
                     └─────────────────┼─────────────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │   Meta-Learner: Logistic Reg      │
                     └───────────────────────────────────┘
```

#### 5.3 Stacked Model 3

```
              ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
              │     KNN     │   │     SVM     │   │     RF      │
              │    k=5      │   │    RBF      │   │   n=100     │
              └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                     │                 │                 │
                     └─────────────────┼─────────────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │   Meta-Learner: Logistic Reg      │
                     └───────────────────────────────────┘
```

### 5.4 Existing Research Stacked Ensemble Architecture

These stacked ensemble configurations are based on existing research literature for comparison with our proposed models.

#### 5.4.1 ER-Stacked Model 1: RF + SVM + GB + DT + KNN

**Source:** Existing research on ensemble deep learning for predictive maintenance

```
              ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
              │     RF      │   │     SVM     │   │     GB      │
              │   n=100     │   │    RBF      │   │   n=100     │
              └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                     │                 │                 │
              ┌──────┴──────┐   ┌──────┴──────┐
              │     DT      │   │     KNN     │
              │             │   │    k=5      │
              └──────┬──────┘   └──────┬──────┘
                     │                 │
                     └─────────────────┼─────────────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │   Meta-Learner: Logistic Reg      │
                     └───────────────────────────────────┘
```

#### 5.4.2 ER-Stacked Model 2: LR + SVM + LDA + GB + AdaBoost

**Source:** Existing research combining linear and boosting methods

```
              ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
              │     LR      │   │     SVM     │   │     LDA     │
              │             │   │    RBF      │   │             │
              └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                     │                 │                 │
              ┌──────┴──────┐   ┌──────┴──────┐
              │     GB      │   │  AdaBoost   │
              │   n=100     │   │   n=100     │
              └──────┬──────┘   └──────┬──────┘
                     │                 │
                     └─────────────────┼─────────────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │   Meta-Learner: Logistic Reg      │
                     └───────────────────────────────────┘
```

#### 5.4.3 ER-Stacked Model 3: All 9 Classifiers

**Source:** Comprehensive ensemble using all available classifiers

```
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │  LR   │ │  KNN  │ │  SVM  │ │  LDA  │ │  GB   │
    └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
        │         │         │         │         │
    ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
    │AdaBoost│ │  DT   │ │  RF   │ │  GNB  │
    └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
        │         │         │         │
        └─────────┴─────────┴─────────┘
                      │
        ┌─────────────▼─────────────┐
        │  Meta-Learner: Logistic Reg │
        └───────────────────────────┘
```

### 5.5 Comparison Framework

| Category | Model | Base Classifiers | Count |
|----------|-------|------------------|-------|
| **Proposed** | Stacked Model 1 | KNN + SVM + RF + AdaBoost + XGBoost | 5 |
| **Proposed** | Stacked Model 2 | RF + XGBoost + AdaBoost | 3 |
| **Proposed** | Stacked Model 3 | KNN + SVM + RF | 3 |
| **Existing Research** | ER-Stacked Model 1 | RF + SVM + GB + DT + KNN | 5 |
| **Existing Research** | ER-Stacked Model 2 | LR + SVM + LDA + GB + AdaBoost | 5 |
| **Existing Research** | ER-Stacked Model 3 | LR + KNN + SVM + LDA + GB + AdaBoost + DT + RF + GNB | 9 |

### 5.6 Dynamic Classifier Selection

**Purpose:** Automatically select optimal classifier combinations for stacking based on research-proven techniques.

#### 5.6.1 Selection Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **Performance-Based** | Select top-k classifiers by individual accuracy | When individual performance is priority |
| **Diversity-Based** | Select classifiers with low Q-statistic (high diversity) | When ensemble diversity is priority |
| **Combined** | Weighted combination of performance and diversity | Balanced approach (recommended) |
| **Greedy Forward** | Iteratively add classifiers that improve ensemble | Most accurate but computationally expensive |

#### 5.6.2 Q-Statistic (Diversity Measure)

The Q-statistic measures pairwise agreement between classifiers:

```
Q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)

Where:
- n11 = Both classifiers correct
- n00 = Both classifiers wrong
- n10 = Classifier 1 correct, Classifier 2 wrong
- n01 = Classifier 1 wrong, Classifier 2 correct

Q ranges from -1 to 1:
- Q = 1: Classifiers always agree (low diversity)
- Q = 0: Classifiers are independent
- Q = -1: Classifiers always disagree (high diversity)
```

#### 5.6.3 Combined Selection Algorithm

```python
# Combined Score = performance_weight * accuracy + (1 - performance_weight) * (1 - avg_Q)

def select_combined(classifiers, y_true, top_k=5, performance_weight=0.6):
    selected = [best_performing_classifier]
    
    while len(selected) < top_k:
        for candidate in remaining_classifiers:
            perf_score = accuracy[candidate]
            avg_q = mean([Q(candidate, s) for s in selected])
            div_score = 1 - avg_q
            combined = performance_weight * perf_score + (1 - performance_weight) * div_score
        
        selected.append(best_candidate)
    
    return selected
```

#### 5.6.4 Implementation

```python
class DynamicClassifierSelector:
    def __init__(self, cv=5, random_state=42):
        self.classifiers = {
            'LR', 'KNN', 'LDA', 'GNB', 'SVM', 
            'DT', 'RF', 'AdaBoost', 'GB', 'XGBoost'
        }
    
    def select_by_performance(self, top_k) -> list
    def select_by_diversity(self, y_true, top_k) -> list
    def select_combined(self, y_true, top_k, performance_weight) -> list
    def greedy_forward_selection(self, X_train, y_train, X_val, y_val, max_k) -> list
    def create_dynamic_stack(self, selected_classifiers) -> StackingClassifier
    def plot_diversity_heatmap() -> None
```

**Dependencies:** sklearn.base.clone, sklearn.ensemble.StackingClassifier

### 6. Data Flow Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VEHMS Data Pipeline                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Step 1: Data Loading                                                         │
│  ┌─────────────┐                                                              │
│  │ CSV File    │──▶ DataLoader.load_dataset() ──▶ pd.DataFrame (10000, 10)   │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 2: Data Visualization                                                   │
│  ┌─────────────┐                                                              │
│  │ DataFrame   │──▶ DataVisualizer.plot_*() ──▶ Inline Plots                 │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 3: Data Cleaning                                                        │
│  ┌─────────────┐                                                              │
│  │ DataFrame   │──▶ DataCleaner.clean() ──▶ Cleaned DataFrame                │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 4: Preprocessing                                                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐                     │
│  │ Cleaned DF  │──▶ │ StandardScaler│──▶ │ LabelEncoder │──▶ X, y arrays     │
│  └─────────────┘    └──────────────┘    └──────────────┘                     │
│                                                                               │
│  Step 5: Train-Test Split                                                     │
│  ┌─────────────┐                                                              │
│  │ X, y arrays │──▶ train_test_split(stratify=y) ──▶ X_train, X_test,        │
│  └─────────────┘                                      y_train, y_test         │
│                                                                               │
│  Step 6: Feature Extraction                                                   │
│  ┌─────────────┐                                                              │
│  │ X_train     │──▶ FeatureExtractor.calculate_importance() ──▶ Rankings     │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 7: Base Classifier Training                                             │
│  ┌─────────────┐                                                              │
│  │ X_train,    │──▶ BaseClassifierModule.train_all() ──▶ 5 Trained Models    │
│  │ y_train     │    (KNN, SVM, RF, AdaBoost, XGBoost)                        │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 8: Stacked Ensemble Training                                            │
│  ┌─────────────┐                                                              │
│  │ X_train,    │──▶ StackedEnsemble.train_stacked_model() ──▶ 3 Stacked      │
│  │ y_train     │                                               Models         │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 9: Model Evaluation                                                     │
│  ┌─────────────┐                                                              │
│  │ All Models  │──▶ ModelEvaluator.evaluate_model() ──▶ Metrics DataFrame    │
│  │ X_test,     │    (RMSE, MAE, Accuracy, Precision, AUC)                    │
│  │ y_test      │                                                              │
│  └─────────────┘                                                              │
│                                                                               │
│  Step 10: XAI Explanations                                                    │
│  ┌─────────────┐                                                              │
│  │ Best Model  │──▶ XAIExplainer.explain() ──▶ SHAP + LIME Visualizations    │
│  │ X_test      │                                                              │
│  └─────────────┘                                                              │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7. Technology Stack

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Core** | Python | 3.9+ | Programming language |
| **Data Processing** | pandas | 2.0+ | Data manipulation |
| | numpy | 1.24+ | Numerical operations |
| | scipy | 1.10+ | Statistical functions |
| **Visualization** | matplotlib | 3.7+ | Static plots |
| | seaborn | 0.12+ | Statistical visualizations |
| | plotly | 5.14+ | Interactive plots |
| **Machine Learning** | scikit-learn | 1.2+ | ML algorithms, preprocessing |
| | xgboost | 1.7+ | Gradient boosting |
| **Explainable AI** | shap | 0.41+ | SHAP explanations |
| | lime | 0.2+ | LIME explanations |
| **Environment** | Jupyter | 1.0+ | Interactive notebook |
| | ipywidgets | 8.0+ | Interactive widgets |

### 8. Performance Targets

| Metric | Target Value | Tolerance |
|--------|--------------|-----------|
| AUC (Stacked Model 1) | 0.9702 | ±0.02 |
| RMSE (Stacked Model 1) | 0.3355 | ±0.05 |
| Accuracy (Stacked Model 1) | 0.9470 | ±0.02 |
| Precision (Stacked Model 1) | 0.9486 | ±0.02 |

### 9. Jupyter Notebook Structure

```
VEHMS_Predictive_Maintenance.ipynb
│
├── 1. Introduction & Setup
│   ├── 1.1 Import Libraries
│   ├── 1.2 Configuration Settings
│   └── 1.3 Helper Functions
│
├── 2. Dataset Loading & Validation
│   ├── 2.1 Load Dataset
│   ├── 2.2 Validate Structure
│   └── 2.3 Display Summary Statistics
│
├── 3. Exploratory Data Analysis
│   ├── 3.1 Distribution Plots
│   ├── 3.2 Correlation Heatmap
│   ├── 3.3 Class Distribution
│   ├── 3.4 Box Plots
│   └── 3.5 Pairwise Plots
│
├── 4. Data Cleaning
│   ├── 4.1 Missing Value Analysis
│   ├── 4.2 Duplicate Detection
│   ├── 4.3 Outlier Handling
│   └── 4.4 Cleaning Summary
│
├── 5. Data Preprocessing
│   ├── 5.1 Feature-Target Separation
│   ├── 5.2 Standard Scaling
│   ├── 5.3 Label Encoding
│   └── 5.4 Train-Test Split
│
├── 6. Feature Extraction
│   ├── 6.1 Calculate Importance
│   ├── 6.2 Rank Features
│   └── 6.3 Visualize Importance
│
├── 7. Base Classifier Training
│   ├── 7.1 Train KNN
│   ├── 7.2 Train SVM
│   ├── 7.3 Train Random Forest
│   ├── 7.4 Train AdaBoost
│   ├── 7.5 Train XGBoost
│   └── 7.6 Compare Base Classifiers
│
├── 8. Stacked Ensemble Models
│   ├── 8.1 Create Stacked Model 1
│   ├── 8.2 Create Stacked Model 2
│   ├── 8.3 Create Stacked Model 3
│   └── 8.4 Train All Stacked Models
│
├── 9. Cross-Validation Evaluation
│   ├── 9.1 K-Fold CV Setup
│   ├── 9.2 Evaluate All Models
│   └── 9.3 CV Results Summary
│
├── 10. Comprehensive Metrics
│   ├── 10.1 Calculate All Metrics
│   ├── 10.2 Generate Confusion Matrices
│   └── 10.3 Verify Target Performance
│
├── 11. Performance Visualization
│   ├── 11.1 Accuracy Comparison
│   ├── 11.2 RMSE Comparison
│   ├── 11.3 AUC Comparison
│   ├── 11.4 ROC Curves
│   └── 11.5 Confusion Matrix Heatmaps
│
├── 12. Final Predictions
│   ├── 12.1 Select Best Model
│   ├── 12.2 Generate Predictions
│   ├── 12.3 Classification Report
│   └── 12.4 Sample Predictions
│
├── 13. Explainable AI (XAI)
│   ├── 13.1 SHAP Analysis
│   │   ├── 13.1.1 Summary Plot
│   │   ├── 13.1.2 Bar Plot
│   │   ├── 13.1.3 Force Plots
│   │   ├── 13.1.4 Waterfall Plots
│   │   └── 13.1.5 Dependence Plots
│   ├── 13.2 LIME Analysis
│   │   ├── 13.2.1 Initialize Explainer
│   │   ├── 13.2.2 Sample Explanations
│   │   └── 13.2.3 Prediction Probabilities
│   ├── 13.3 Combined Analysis
│   │   ├── 13.3.1 SHAP vs LIME Comparison
│   │   ├── 13.3.2 Natural Language Explanations
│   │   └── 13.3.3 Class Influence Analysis
│   └── 13.4 XAI Summary
│
├── 14. Prediction Function
│   ├── 14.1 Input Validation
│   ├── 14.2 Prediction with Confidence
│   └── 14.3 Example Usage
│
└── 15. Conclusions
    ├── 15.1 Results Summary
    ├── 15.2 Performance vs Targets
    └── 15.3 Recommendations
```

### 10. Error Handling Strategy

```python
class VEHMSException(Exception):
    """Base exception for VEHMS"""
    pass

class DataValidationError(VEHMSException):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(VEHMSException):
    """Raised when model training fails"""
    pass

class PredictionError(VEHMSException):
    """Raised when prediction fails"""
    pass

# Error handling patterns
def safe_load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        validate_dataframe(df)
        return df
    except FileNotFoundError:
        raise DataValidationError(f"Dataset file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise DataValidationError("Dataset file is empty")
    except Exception as e:
        raise DataValidationError(f"Failed to load dataset: {str(e)}")

def safe_predict(model, X: np.ndarray) -> tuple:
    try:
        prediction = model.predict(X)
        proba = model.predict_proba(X)
        return prediction, proba
    except Exception as e:
        raise PredictionError(f"Prediction failed: {str(e)}")
```

### 11. Testing Strategy

| Test Type | Scope | Tools |
|-----------|-------|-------|
| Unit Tests | Individual components | pytest |
| Integration Tests | Component interactions | pytest |
| Data Validation | Input/output data | pandas testing |
| Model Validation | Performance metrics | sklearn metrics |
| Notebook Execution | End-to-end workflow | nbconvert |

### 12. Requirements Traceability

| Requirement | Design Component | Implementation |
|-------------|------------------|----------------|
| Req 1: Dataset Loading | DataLoader | Section 4.1 |
| Req 2: EDA Visualization | DataVisualizer | Section 4.2 |
| Req 3: Data Cleaning | DataCleaner | Section 4.3 |
| Req 4: Preprocessing | DataPreprocessor | Section 4.4 |
| Req 5: Feature Extraction | FeatureExtractor | Section 4.5 |
| Req 6: Base Classifiers | BaseClassifierModule | Section 4.6 |
| Req 7: Stacked Ensemble | StackedEnsemble | Section 4.7, 5 |
| Req 8: Cross-Validation | ModelEvaluator | Section 4.8 |
| Req 9: Performance Metrics | ModelEvaluator | Section 4.8 |
| Req 10: Visualization | DataVisualizer | Section 4.2 |
| Req 11: Final Predictions | All components | Section 6 |
| Req 12: XAI (SHAP + LIME) | XAIExplainer | Section 4.9 |
| Req 13: Input Validation | DataLoader | Section 4.1 |
| Req 14: Jupyter Notebook | All components | Section 9 |
