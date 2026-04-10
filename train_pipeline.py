import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

os.makedirs('dataset', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('model', exist_ok=True)

dataset_path = 'dataset/ObesityDataSet.csv'

print("Generating synthetic dataset with correct BMI classes...")
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
}
df = pd.DataFrame(data)

# Calculate accurate BMI to enforce strict mathematical relation for the target labels
bmi = df['Weight'] / (df['Height'] ** 2)
conditions = [
    (bmi < 18.5),
    (bmi >= 18.5) & (bmi < 25),
    (bmi >= 25) & (bmi < 27.5),
    (bmi >= 27.5) & (bmi < 30),
    (bmi >= 30) & (bmi < 35),
    (bmi >= 35) & (bmi < 40),
    (bmi >= 40)
]
choices = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]
df['NObeyesdad'] = np.select(conditions, choices, default='Normal_Weight')

df.to_csv(dataset_path, index=False)

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X['BMI_calculated'] = X['Weight'] / (X['Height'] ** 2)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'model/label_encoder.pkl')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, 'model/preprocessor.pkl')

feature_names = numerical_cols + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    results[name] = {'Model': model, 'Accuracy': accuracy_score(y_test, y_pred)}

best_model_name = pd.DataFrame(results).T['Accuracy'].astype(float).idxmax()
best_model = results[best_model_name]['Model']

print(f"Best model found: {best_model_name}")

joblib.dump(best_model, 'model/best_model.pkl')
print("Model, scaler, and encoder saved to model/")
