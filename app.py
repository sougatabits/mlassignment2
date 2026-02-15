import os
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


st.set_page_config(page_title="Rainfall Prediction App (Australia)", layout="wide")
st.title("Rainfall Prediction App (Australia)")
st.write("Machine Learning Assignment 2 - Submitted by: Sougata Das BITSID:2024DC04257")


MODEL_FOLDER = "models"

feature_names = load_pickle(f"{MODEL_FOLDER}/feature_names.pkl")
scaler = load_pickle(f"{MODEL_FOLDER}/scaler.pkl")
encoders = load_pickle(f"{MODEL_FOLDER}/encoders.pkl")

model_files = {
    "Logistic Regression": "rainfall_logistic_regression_classification.pkl",
    "Decision Tree": "rainfall_decision_tree_classification.pkl",
    "K-Nearest Neighbor": "rainfall_k-nearest_neighbor_classification.pkl",
    "Naive Bayes": "rainfall_naive_bayes_classification.pkl",
    "Random Forest": "rainfall_random_forest_classification.pkl",
    "XGBoost": "xgboost.pkl"
}

models = {name: load_pickle(f"{MODEL_FOLDER}/{file}") for name, file in model_files.items()}


st.sidebar.header("Manual Input Features")

def user_input_features():
    feature_values = {}
    essential_features = ["Month","Location","Temp_range", "Humidity_avg", "Pressure_avg"]

    for feature in essential_features:
        feature_values[feature] = st.sidebar.number_input(feature)

    df = pd.DataFrame([feature_values])

    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]
    return df

input_df = user_input_features()
input_scaled = scaler.transform(input_df)

selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]

st.subheader("Prediction for Manual Input")
rain_prediction = model.predict(input_scaled)[0]

if hasattr(model, "predict_proba"):
    rain_prob = model.predict_proba(input_scaled)[0][1]
else:
    rain_prob = None

st.write("Rain Tomorrow? →", "Yes!! there is a possibility of rain tomorrow" if rain_prediction == 1 else "No Rain Tomorrow")
if rain_prob is not None:
    st.write(f"Probability of rain: {rain_prob*100:.2f}%")

st.write("---")
st.write("**Input Features Provided:**")
st.dataframe(input_df)


st.sidebar.header("Upload Test Dataset (CSV)")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()  # clean column names

    if "RainTomorrow" not in test_df.columns:
        st.error("Uploaded CSV must contain 'RainTomorrow' column")
    else:
        X_test = test_df.drop("RainTomorrow", axis=1)
        y_test = test_df["RainTomorrow"]

        # Encode categorical columns using saved encoders
        for col, le in encoders.items():
            if col in X_test.columns:
                X_test[col] = X_test[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Add missing columns to match training features
        for col in feature_names:
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_test = X_test[feature_names]

        # Scale features
        X_test_scaled = scaler.transform(X_test)

        st.subheader("Uploaded Test Dataset Predictions")

        # Model selection for uploaded CSV
        selected_model_csv = st.sidebar.selectbox(
            "Choose Model for Test CSV",
            list(models.keys()),
            key="csv_model"
        )
        model_csv = models[selected_model_csv]

        y_pred = model_csv.predict(X_test_scaled)

        # Encode target for evaluation
        le_target = encoders.get("RainTomorrow", None)
        if le_target:
            y_test_encoded = y_test.map(
                lambda x: le_target.transform([x])[0] if x in le_target.classes_ else -1
            )
        else:
            y_test_encoded = y_test

        # -----------------------------
        # Display Predictions Table
        # -----------------------------
        results_df = test_df.copy()
        results_df["Predicted_RainTomorrow"] = y_pred
        st.subheader("Predictions Table")
        st.dataframe(results_df)

        metrics_dict = {
            "Accuracy": [accuracy_score(y_test_encoded, y_pred)],
            "Precision": [precision_score(y_test_encoded, y_pred)],
            "Recall": [recall_score(y_test_encoded, y_pred)],
            "F1-Score": [f1_score(y_test_encoded, y_pred)]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        st.subheader("Evaluation Metrics")
        st.table(metrics_df)

        cm = confusion_matrix(y_test_encoded, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {cls}" for cls in np.unique(y_test_encoded)],
            columns=[f"Predicted {cls}" for cls in np.unique(y_test_encoded)]
        )
        st.subheader("Confusion Matrix")
        st.table(cm_df)
 
