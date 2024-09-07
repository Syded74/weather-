import joblib
import pandas as pd
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models and scaler
rf_model = joblib.load(os.path.join(current_dir, 'random_forest_model.joblib'))
gb_model = joblib.load(os.path.join(current_dir, 'gradient_boosting_model.joblib'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))

def prepare_input(data, scaler):
    features = data.drop(columns=['Date', 'Location'], errors='ignore')
    scaled_features = scaler.transform(features)
    return scaled_features

def make_prediction(data):
    X = prepare_input(data, scaler)
    rf_pred = rf_model.predict(X)
    gb_pred = gb_model.predict(X)
    return rf_pred + gb_pred