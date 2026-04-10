from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import traceback
# Reload trigger comment

app = Flask(__name__)

# Load models and transformers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

try:
    best_model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
except Exception as e:
    print("Error loading models. Have you run `train_pipeline.py`?")
    best_model = None

# Health Suggestions mapped to output labels
def get_health_suggestion(obesity_level):
    suggestions = {
        'Insufficient_Weight': "You are underweight. Consider consulting a nutritionist to increase your intake of nutrient-dense foods, proteins, and healthy fats.",
        'Normal_Weight': "Maintain your current healthy lifestyle! Continue eating a balanced diet and regular physical activity.",
        'Overweight_Level_I': "You are slightly overweight. Try to reduce portion sizes slightly and increase daily steps or moderate activities.",
        'Overweight_Level_II': "Consider incorporating more cardiovascular exercises and tracking your caloric intake to prevent further weight gain.",
        'Obesity_Type_I': "Focus on a structured weight loss plan. Increase physical activity (e.g., 30-45 mins daily) and seek dietetic advice.",
        'Obesity_Type_II': "It is highly recommended to consult a healthcare professional. A strict balance of diet and professional exercise routines is needed.",
        'Obesity_Type_III': "Severe obesity detected. Please seek immediate medical counseling for comprehensive weight management interventions."
    }
    return suggestions.get(obesity_level, "No specific suggestion available.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not best_model:
        return jsonify({'error': 'Prediction model not loaded on server.'}), 500

    try:
        data = request.json
        print("Received Data:", data)
        
        # Extract features and handle type conversion safely
        row = {
            'Gender': data.get('Gender', 'Male'),
            'Age': float(data.get('Age', 25)),
            'Height': float(data.get('Height', 1.70)),
            'Weight': float(data.get('Weight', 70)),
            'family_history_with_overweight': data.get('family_history_with_overweight', 'no'),
            'FAVC': data.get('FAVC', 'no'),
            'FCVC': float(data.get('FCVC', 2.0)),
            'NCP': float(data.get('NCP', 3.0)),
            'CAEC': data.get('CAEC', 'Sometimes'),
            'SMOKE': data.get('SMOKE', 'no'),
            'CH2O': float(data.get('CH2O', 2.0)),
            'SCC': data.get('SCC', 'no'),
            'FAF': float(data.get('FAF', 0.0)),
            'TUE': float(data.get('TUE', 0.0)),
            'CALC': data.get('CALC', 'no'),
            'MTRANS': data.get('MTRANS', 'Public_Transportation')
        }
        
        # Explicit feature engineering (BMI) computed exactly as in training
        row['BMI_calculated'] = row['Weight'] / (row['Height'] ** 2)
        print("Processed Row:", row)
        
        input_df = pd.DataFrame([row])
        
        # Preprocessing pipeline
        X_processed = preprocessor.transform(input_df)
        feature_names = preprocessor.transformers_[0][2] + preprocessor.named_transformers_['cat'].get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
        
        # Provide correct feature names to prevent XGBoost warnings
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Prediction
        pred_encoded = best_model.predict(X_processed_df)
        pred_class = label_encoder.inverse_transform(pred_encoded)[0]
        
        # Build Response
        response = {
            'obesity_level': pred_class,
            'suggestion': get_health_suggestion(pred_class),
            'calculated_bmi': round(row['BMI_calculated'], 2)
        }
        
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
