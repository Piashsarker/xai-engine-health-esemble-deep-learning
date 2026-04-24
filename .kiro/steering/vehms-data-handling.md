# VEHMS Data Handling Guide

---
inclusion: fileMatch
fileMatchPattern: "*.csv,*data*,*dataset*,*loader*"
---

## Overview

Guidelines for handling data in the VEHMS predictive maintenance system, including loading, preprocessing, and feature engineering.

---

## 1. Dataset Schema

### Required Columns

| Column | Type | Description | Expected Range |
|--------|------|-------------|----------------|
| Crankshaft | float64 | Crankshaft sensor reading | 4.0 - 21.0 |
| Overheating | float64 | Overheating indicator | 8.0 - 26.0 |
| Lubricant | float64 | Lubricant level/quality | 4.0 - 16.0 |
| Misfires | float64 | Engine misfire intensity | 2.5 - 11.0 |
| Piston | float64 | Piston sensor reading | 3.5 - 16.0 |
| Starter | float64 | Starter motor measurement | 250.0 - 450.0 |
| Temperature | float64 | Environmental temperature (°C) | 15.0 - 35.0 |
| Humidity | float64 | Environmental humidity (%) | 30.0 - 90.0 |
| Altitude | float64 | Environmental altitude (m) | 0.0 - 3000.0 |
| Decision | categorical | Target variable | Good, Minor, Moderate, Critical |

### Feature Categories

- **Sensor Features**: Crankshaft, Overheating, Lubricant, Misfires, Piston, Starter
- **Environmental Features**: Temperature, Humidity, Altitude
- **Target Variable**: Decision

---

## 2. Loading Different Datasets

### Standard CSV Loading
```python
df = pd.read_csv('dataset/augmented_data_with_environment.csv')
```

### Loading with Validation
```python
loader = DataLoader()
df = loader.load_dataset('path/to/dataset.csv')
loader.validate_columns(df)
loader.validate_data_types(df)
```

### Loading from Different Sources

```python
# From URL
df = pd.read_csv('https://example.com/data.csv')

# From Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# From JSON
df = pd.read_json('data.json')

# From SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM engine_data', conn)
```

---

## 3. Data Preprocessing Options

### Scaling Methods

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (default) - zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler - scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RobustScaler - robust to outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Encoding Options

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# LabelEncoder (default) - alphabetical order
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Result: Critical=0, Good=1, Minor=2, Moderate=3

# Custom order encoding
order_map = {'Good': 0, 'Minor': 1, 'Moderate': 2, 'Critical': 3}
y_encoded = y.map(order_map)
```

### Handling Missing Values

```python
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()

# Fill with mean
df['column'] = df['column'].fillna(df['column'].mean())

# Fill with median (robust to outliers)
df['column'] = df['column'].fillna(df['column'].median())

# Fill with mode (for categorical)
df['column'] = df['column'].fillna(df['column'].mode()[0])

# Forward/backward fill (time series)
df['column'] = df['column'].fillna(method='ffill')
```

### Handling Outliers

```python
# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Cap outliers
df['column'] = df['column'].clip(lower_bound, upper_bound)

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df[numeric_columns]))
df_clean = df[(z_scores < 3).all(axis=1)]
```

---

## 4. Feature Engineering

### Creating New Features

```python
# Interaction features
df['Crankshaft_x_Overheating'] = df['Crankshaft'] * df['Overheating']

# Ratio features
df['Lubricant_Piston_Ratio'] = df['Lubricant'] / (df['Piston'] + 1)

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Binning
df['Temp_Category'] = pd.cut(df['Temperature'], bins=[0, 20, 25, 30, 40], 
                              labels=['Cold', 'Normal', 'Warm', 'Hot'])
```

### Feature Selection

```python
# Variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
selector = RFE(RandomForestClassifier(), n_features_to_select=5)
X_selected = selector.fit_transform(X, y)

# Feature importance from model
rf = RandomForestClassifier().fit(X, y)
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 5. Data Augmentation

### Synthetic Data Generation

```python
# SMOTE for imbalanced classes
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Adding noise
noise = np.random.normal(0, 0.1, X.shape)
X_augmented = X + noise
```

---

## 6. Data Validation Checklist

Before training:

- [ ] All required columns present
- [ ] No missing values (or handled appropriately)
- [ ] Numeric columns are float/int type
- [ ] Target column has valid categories only
- [ ] No duplicate rows (or handled)
- [ ] Outliers identified and handled
- [ ] Features scaled appropriately
- [ ] Target encoded correctly
- [ ] Train/test split is stratified
- [ ] Data shapes are correct

```python
# Quick validation
def validate_data(X_train, X_test, y_train, y_test):
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_train distribution: {np.bincount(y_train)}")
    print(f"y_test distribution: {np.bincount(y_test)}")
    print(f"Any NaN in X_train: {np.isnan(X_train).any()}")
    print(f"Any NaN in X_test: {np.isnan(X_test).any()}")
```
