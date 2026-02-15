import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
 
#load and pre-process dataset
def load_and_preprocess_data(filepath="data/weatherAUS.csv"):
    df = pd.read_csv(filepath)
 
    df = df.dropna(subset=["RainTomorrow"])
 
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
 
    if "Cloud9am" in df.columns and "Cloud3pm" in df.columns:
        df["Cloud_diff"] = df["Cloud3pm"] - df["Cloud9am"]
        df["Cloud_avg"] = (df["Cloud3pm"] + df["Cloud9am"]) / 2
 
    if "MinTemp" in df.columns and "MaxTemp" in df.columns:
        df["Temp_range"] = df["MaxTemp"] - df["MinTemp"]
 
    if "Humidity9am" in df.columns and "Humidity3pm" in df.columns:
        df["Humidity_diff"] = df["Humidity3pm"] - df["Humidity9am"]
        df["Humidity_avg"] = (df["Humidity3pm"] + df["Humidity9am"]) / 2
 
    if "Pressure9am" in df.columns and "Pressure3pm" in df.columns:
        df["Pressure_avg"] = (df["Pressure3pm"] + df["Pressure9am"]) / 2
        
 
    cols_to_drop = ["Date","Cloud9am", "Cloud3pm", "MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm","Pressure9am","Pressure3pm"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
 
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    print(df.head())
    print(df.columns)
 
    le_target = LabelEncoder()
    df["RainTomorrow"] = le_target.fit_transform(df["RainTomorrow"])
 
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
 
    X = df.drop("RainTomorrow", axis=1)
    y = df["RainTomorrow"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    print("Data Loaded !")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X_train.columns)
#model training
def train_and_save_models():
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
 
    os.makedirs("models", exist_ok=True)
 
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
 
    models = {
        "rainfall_logistic_regression_classification": LogisticRegression(max_iter=2000),
        "rainfall_decision_tree_classification": DecisionTreeClassifier(random_state=42),
        "rainfall_k-nearest_neighbor_classification": KNeighborsClassifier(),
        "rainfall_naive_bayes_classification": GaussianNB(),
        "rainfall_random_forest_classification": RandomForestClassifier(random_state=42),
        "xgboost": xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    }
 
    results = []
 
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 
        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred  # fallback
 
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)
 
        # Save model
        model_file = f"models/{name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
 
    # Save metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv("comp_matrix_assignment.csv", index=False)
    print("Training completed! Metrics saved to comp_matrix_assignment.csv")
    print(results_df)
 
if __name__ == "__main__":
    train_and_save_models()
