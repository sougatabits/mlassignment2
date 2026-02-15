import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Rainfall Prediction App -  (Australia)", layout="wide")
st.title("Rainfall Prediction App (Australia)")
st.write("Machine Learning Assignment 2 -  Submitted by: Sougata Das BITSID:2024DC04257")

MODEL_FOLDER = "models"

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load feature names and scaler
feature_names = load_pickle(f"models/feature_names.pkl")
scaler = load_pickle(f"models/scaler.pkl")

# Load trained models
model_files = {
    "Logistic Regression": "rainfall_logistic_regression_classification.pkl",
    "Decision Tree": "rainfall_decision_tree_classification.pkl",
    "K-Nearest Neighbor": "rainfall_k-nearest_neighbor_classification.pkl",
    "Naive Bayes": "rainfall_naive_bayes_classification.pkl",
    "Random Forest": "rainfall_random_forest_classification.pkl",
    "XGBoost": "xgboost.pkl"
}

models = {name: load_pickle(f"models/{file}") for name, file in model_files.items()}

st.sidebar.header("Input Features (Minimal)")

def user_input_features():
    feature_values = {}
    # Only essential features for user input
    essential_features = ["Month","Location","Temp_range", "Humidity", "Pressure_avg"]

    for feature in essential_features:
        feature_values[feature] = st.sidebar.number_input(feature)

    df = pd.DataFrame([feature_values])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]
    return df

input_df = user_input_features()
input_scaled = scaler.transform(input_df)

#predict
st.subheader("Prediction")
selected_model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[selected_model_name]

rain_prediction = model.predict(input_scaled)[0]

if hasattr(model, "predict_proba"):
    tmrw_rain_probability = model.predict_proba(input_scaled)[0][1]
else:
    tmrw_rain_probability = None

st.write("Rain Tomorrow? →", "Yes!! there is a possibility of rain tommorrow" if rain_prediction == 1 else "No Rain Tommorrow")
if tmrw_rain_probability is not None:
    st.write(f"Probability of rain: {tmrw_rain_probability*100:.2f}%")

st.write("---")
st.write("**Input Features Provided:**")
st.dataframe(input_df)

 
