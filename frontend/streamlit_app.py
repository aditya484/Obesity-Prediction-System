import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🧬", layout="centered")

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

@st.cache_resource
def load_models():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return model, preprocessor, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure you ran train_pipeline.py first.")
        return None, None, None

model, preprocessor, label_encoder = load_models()

def get_health_suggestion(obesity_level):
    suggestions = {
        'Insufficient_Weight': "You are underweight. Consider consulting a nutritionist to increase nutrient intake.",
        'Normal_Weight': "Great job! Maintain your current healthy lifestyle with a balanced diet and regular physical activity.",
        'Overweight_Level_I': "You are slightly overweight. Try reducing portion sizes and increasing daily steps.",
        'Overweight_Level_II': "Incorporate more cardiovascular exercises and track caloric intake to prevent weight gain.",
        'Obesity_Type_I': "Focus on a structured weight loss plan. Increase physical activity and seek dietetic advice.",
        'Obesity_Type_II': "Consult a healthcare professional. A strict balance of diet and exercise routines is advised.",
        'Obesity_Type_III': "Severe obesity detected. Seek medical counseling for comprehensive weight management."
    }
    return suggestions.get(obesity_level, "No specific suggestion available.")

st.title("🧬 Obesity Risk Prediction System")
st.write("Enter your daily habits and physical condition to analyze your obesity risk.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (Years)", 10.0, 100.0, 25.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", 1.0, 2.5, 1.70)
        weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
        family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
        favc = st.selectbox("Frequent High Caloric Food", ["yes", "no"])
        fcvc = st.slider("Frequency of Vegetables (1-3)", 1.0, 3.0, 2.0)
        ncp = st.slider("Number of Main Meals (1-4)", 1.0, 4.0, 3.0)
        
    with col2:
        caec = st.selectbox("Food Between Meals", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Smoker?", ["no", "yes"])
        ch2o = st.slider("Daily Water Consumption (1-3 L)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Calorie Monitoring", ["no", "yes"])
        faf = st.slider("Physical Activity Frequency (0-3 days)", 0.0, 3.0, 1.0)
        tue = st.slider("Time using Tech Devices (0-2)", 0.0, 2.0, 1.0)
        calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
        
    submit_button = st.form_submit_button("Predict Obesity Level")

if submit_button:
    if model:
        # Calculate BMI
        bmi = weight / (height ** 2)
        
        row = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': family_history,
            'FAVC': favc,
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': caec,
            'SMOKE': smoke,
            'CH2O': ch2o,
            'SCC': scc,
            'FAF': faf,
            'TUE': tue,
            'CALC': calc,
            'MTRANS': mtrans,
            'BMI_calculated': bmi
        }])
        
        # Preprocess
        X_processed = preprocessor.transform(row)
        cat_features = preprocessor.transformers_[1][2]
        feature_names = preprocessor.transformers_[0][2] + preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
        
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Predict
        pred_encoded = model.predict(X_df)
        pred_class = label_encoder.inverse_transform(pred_encoded)[0]
        formatted_class = pred_class.replace('_', ' ')
        
        st.divider()
        st.subheader("Results")
        st.write(f"**Predicted Category:** {formatted_class}")
        st.write(f"**Estimated BMI:** {bmi:.2f}")
        st.info(f"**Health Recommendation:** {get_health_suggestion(pred_class)}")
