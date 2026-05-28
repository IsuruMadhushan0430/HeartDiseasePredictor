import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import base64

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted.csv">Download Predictions as CSV</a>'
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])


with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    @st.cache_data
    def get_category_maps():
        data = pd.read_csv("heart.csv")
        cat_cols = data.select_dtypes(include="object").columns
        mappings = {}
        for col in cat_cols:
            uniques = list(data[col].unique())
            mappings[col] = {value: index for index, value in enumerate(uniques)}
        return mappings

    category_maps = get_category_maps()

    sex = category_maps["Sex"]["M" if sex == "Male" else "F"]
    chest_pain_map = {
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
        "Non-Anginal Pain": "NAP",
        "Asymptomatic": "ASY",
    }
    chest_pain = category_maps["ChestPainType"][chest_pain_map[chest_pain]]
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg_map = {
        "Normal": "Normal",
        "ST-T Wave Abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH",
    }
    resting_ecg = category_maps["RestingECG"][resting_ecg_map[resting_ecg]]
    exercise_angina = category_maps["ExerciseAngina"]["Y" if exercise_angina == "Yes" else "N"]
    st_slope_map = {
        "Upsloping": "Up",
        "Flat": "Flat",
        "Downsloping": "Down",
    }
    st_slope = category_maps["ST_Slope"][st_slope_map[st_slope]]

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    algonames = ["Decision Trees", "Logistic Regression", "Random Forest", "Support Vector Machine"]
    modelnames = ["heart_disease_predictor_DecisionTree.pkl", "heart_disease_predictor_LogisticRegression.pkl", "heart_disease_predictor_RandomForest.pkl", "heart_disease_predictor_SVM.pkl"]

    @st.cache_resource
    def load_models():
        models = {}
        for modelname in modelnames:
            model_path = Path(modelname)
            if not model_path.exists() or model_path.stat().st_size == 0:
                models[modelname] = None
                continue
            with model_path.open("rb") as handle:
                models[modelname] = pickle.load(handle)
        return models

    def predict_heart_disease(data):
        models = load_models()
        predictions = []
        errors = []

        for modelname in modelnames:
            model = models.get(modelname)
            if model is None:
                errors.append(f"Model file is missing or empty: {modelname}")
                predictions.append(None)
                continue
            prediction = model.predict(data)
            predictions.append(prediction)

        return predictions, errors
    
    if st.button("Submit"):
        st.subheader("Results....")
        st.markdown("-----------------------------------------")

        result, errors = predict_heart_disease(input_data)

        for message in errors:
            st.error(message)

        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i] is None:
                st.write("Prediction unavailable.")
            elif result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown("-----------------------------------------")

with tab2:
    st.title("Upload CSV File")
    st.subheader("Instructions to note before uploading the files:")
    st.info("""
1. The CSV file should contain the following columns:
    - Age (years)
    - Sex (Male/Female)
    - ChestPainType (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic)
    - RestingBP (mm Hg)
    - Cholesterol (mg/dl)
    - FastingBS (<= 120 mg/dl or > 120 mg/dl)
    - RestingECG (Normal, ST-T Wave Abnormality, Left Ventricular Hypertrophy)
    - MaxHR
    - ExerciseAngina (Yes/No)
    - Oldpeak (ST depression)
    - ST_Slope (Upsloping, Flat, Downsloping)
2. Ensure that the column names match exactly as listed above.
3. The file should be in CSV format and should not exceed 5MB in size.
            """)
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("heart_disease_predictor_LogisticRegression.pkl", "rb"))

        expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

        if set(expected_columns).issubset(input_data.columns):

            input_data["Prediction LR"] = ''

            for i in range(len(input_data)):
                arr = input_data.iloc[i:1-i].values
                input_data["Prediction LR"][i] = model.predict([arr])[0]
            input_data.to_csv("PredictedLR.csv")

            st.subheader("Predictions:")
            st.write(input_data)

            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)

        else:
            st.warning("Please make sure the uploaded CSV files has the correct columns.")

    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px
    data = {'Decision Trees': 86.41, 'Logistic Regression': 86.95, 'Random Forest': 85.32, 'Support Vector Machine': 86.33}
    models = list(data.keys())
    accuracies = list(data.values())
    df = pd.DataFrame(list(zip(models, accuracies)), columns=['models', 'accuracies'])
    fig = px.bar(df, y='accuracies', x='models', title='Model Accuracies')
    st.plotly_chart(fig)