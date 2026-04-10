import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Obesity Prediction System - EDA and Modeling
This notebook covers data generation/loading, preprocessing, Exploratory Data Analysis (EDA), and machine learning model training.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Set aesthetic parameters for seaborn
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""
nb['cells'].append(nbf.v4.new_code_cell(code_imports))

text_data = """\
## 1. Dataset Handling
We check if the dataset exists in `../dataset/`. If not, we generate a synthetic dataset mimicking the UCI Obesity dataset.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text_data))

code_data = """\
dataset_path = '../dataset/ObesityDataSet.csv'

if not os.path.exists(dataset_path):
    print("Dataset not found. Generating synthetic dataset to mimic the UCI Obesity dataset...")
    np.random.seed(42)
    n_samples = 2111
    
    data = {
        'Gender': np.random.choice(['Female', 'Male'], n_samples),
        'Age': np.random.uniform(14, 61, n_samples).round(1),
        'Height': np.random.uniform(1.45, 1.98, n_samples).round(2),
        'Weight': np.random.uniform(39, 173, n_samples).round(1),
        'family_history_with_overweight': np.random.choice(['yes', 'no'], n_samples, p=[0.8, 0.2]),
        'FAVC': np.random.choice(['yes', 'no'], n_samples, p=[0.88, 0.12]),
        'FCVC': np.random.uniform(1, 3, n_samples).round(1),
        'NCP': np.random.uniform(1, 4, n_samples).round(1),
        'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples, p=[0.05, 0.95]),
        'CH2O': np.random.uniform(1, 3, n_samples).round(1),
        'SCC': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
        'FAF': np.random.uniform(0, 3, n_samples).round(1),
        'TUE': np.random.uniform(0, 2, n_samples).round(1),
        'CALC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'], n_samples),
        'NObeyesdad': np.random.choice([
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 
            'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ], n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print("Generated synthetic dataset at:", dataset_path)
else:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")

display(df.head())
"""
nb['cells'].append(nbf.v4.new_code_cell(code_data))

code_eda1 = """\
# Basic info
df.info()
print("\\nMissing values:\\n", df.isnull().sum())
"""
nb['cells'].append(nbf.v4.new_code_cell(code_eda1))

text_eda = """\
## 2. Exploratory Data Analysis (EDA)
Visualizing distributions and correlations.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text_eda))

code_eda2 = """\
# Target Class Distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index, palette='viridis')
plt.title("Distribution of Obesity Levels")
plt.tight_layout()
plt.savefig('../outputs/target_distribution.png')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title("Age Distribution")
plt.savefig('../outputs/age_distribution.png')
plt.show()

# Weight vs Height by Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Height', y='Weight', hue='Gender', style='NObeyesdad', data=df, alpha=0.7)
plt.title("Weight vs Height colored by Gender")
plt.savefig('../outputs/weight_height_scatter.png')
plt.show()

# Correlation Heatmap for numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('../outputs/correlation_heatmap.png')
plt.show()
"""
nb['cells'].append(nbf.v4.new_code_cell(code_eda2))

text_prep = """\
## 3. Data Preprocessing
Encoding categorical features, feature engineering (creating a generic BMI feature), and standardizing numeric values.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text_prep))

code_prep = """\
# Separate target and features
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Feature Engineering: Calculate explicit BMI (Weight / Height^2)
X['BMI_calculated'] = X['Weight'] / (X['Height'] ** 2)

# Encoding Target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, '../model/label_encoder.pkl')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Create Preprocessing Pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit and transform features
X_processed = preprocessor.fit_transform(X)
feature_names = numerical_cols + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

joblib.dump(preprocessor, '../model/preprocessor.pkl')

print("Preprocessing complete. X_processed shape:", X_processed.shape)
"""
nb['cells'].append(nbf.v4.new_code_cell(code_prep))

text_smote = """\
## 4. Train-Test Split & Handling Imbalance
We apply SMOTE to balance the training data if needed.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text_smote))

code_smote = """\
X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Original training shape:", X_train.shape)
print("Resampled training shape:", X_train_resampled.shape)
"""
nb['cells'].append(nbf.v4.new_code_cell(code_smote))

text_models = """\
## 5. Model Training & Comparison
We train Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text_models))

code_models = """\
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'Model': model, 'Accuracy': accuracy, 'F1 Score': f1}

# Display Results
results_df = pd.DataFrame(results).T[['Accuracy', 'F1 Score']]
print("\\nModel Comparison:\\n", results_df.sort_values(by='Accuracy', ascending=False))
"""
nb['cells'].append(nbf.v4.new_code_cell(code_models))

text_eval = """\
## 6. Model Evaluation & Saving
Choose the best model, output detailed metrics, and save it.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text_eval))

code_eval = """\
# Select best model
best_model_name = results_df['Accuracy'].astype(float).idxmax()
best_model = results[best_model_name]['Model']

print(f"\\nBest Model Default: {best_model_name}")

# Detailed Evaluation
y_pred_best = best_model.predict(X_test)
print(f"\\nClassification Report for {best_model_name}:\\n")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/confusion_matrix.png')
plt.show()

# Feature Importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    plt.figure(figsize=(12, 6))
    plt.title("Top Feature Importances")
    plt.bar(range(15), importances[indices], align="center")
    plt.xticks(range(15), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('../outputs/feature_importances.png')
    plt.show()

# Save Best Model
joblib.dump(best_model, '../model/best_model.pkl')
print("\\nBest model saved to `../model/best_model.pkl`")
"""
nb['cells'].append(nbf.v4.new_code_cell(code_eval))

with open('notebooks/Obesity_Prediction_EDA_and_Modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook created successfully at notebooks/Obesity_Prediction_EDA_and_Modeling.ipynb")
