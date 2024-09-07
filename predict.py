import joblib
import pandas as pd

rf_model = joblib.load('random_forest_model.joblib')
gb_model = joblib.load('gradient_boosting_model.joblib')
scaler = joblib.load('scaler.joblib')

def prepare_input(data, scaler):
    features = data.drop(columns=['Date', 'Location'], errors='ignore')
    scaled_features = scaler.transform(features)
    return scaled_features

def make_prediction(data):
    X = prepare_input(data, scaler)
    rf_pred = rf_model.predict(X)
    gb_pred = gb_model.predict(X)
    return rf_pred + gb_pred