"""
model.py - ML logic for Real Estate Price Prediction
Models: Linear Regression + Random Forest
"""

import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES = [
    'year_sold', 'property_tax', 'insurance', 'beds', 'baths',
    'sqft', 'year_built', 'lot_size', 'basement', 'popular',
    'recession', 'property_age', 'property_type_Condo'
]


def load_data(filepath: str):
    """Load and return the real estate dataset."""
    try:
        logging.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def split_data(df: pd.DataFrame, test_size=0.2):
    """Separate features/target and split into train/test sets."""
    try:
        x = df.drop('price', axis=1)
        y = df['price']
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size,
            stratify=x['property_type_Condo'],
            random_state=42
        )
        logging.info(f"Train: {x_train.shape}, Test: {x_test.shape}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Data split failed: {e}")
        raise


def train_linear_regression(x_train, y_train):
    """Train a Linear Regression model."""
    try:
        logging.info("Training Linear Regression...")
        model = LinearRegression()
        lrmodel = model.fit(x_train, y_train)
        logging.info("Linear Regression trained.")
        return lrmodel
    except Exception as e:
        logging.error(f"LR training failed: {e}")
        raise


def train_random_forest(x_train, y_train, n_estimators=200):
    """Train a Random Forest Regressor."""
    try:
        logging.info(f"Training Random Forest (n={n_estimators})...")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion='absolute_error',
            random_state=42
        )
        rfmodel = rf.fit(x_train, y_train)
        logging.info("Random Forest trained.")
        return rfmodel
    except Exception as e:
        logging.error(f"RF training failed: {e}")
        raise


def evaluate_model(model, x_train, y_train, x_test, y_test):
    """Return train and test MAE for a model."""
    try:
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        logging.info(f"Train MAE: ${train_mae:,.0f} | Test MAE: ${test_mae:,.0f}")
        return train_mae, test_mae, train_pred, test_pred
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


def save_model(model, path='RE_Model.pkl'):
    """Save model to disk using pickle."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Save failed: {e}")
        raise


def load_model(path='RE_Model.pkl'):
    """Load model from disk using pickle."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Load failed: {e}")
        raise


def predict_price(model, input_dict: dict, columns) -> float:
    """Make a single price prediction from user input dict."""
    try:
        input_df = pd.DataFrame([input_dict], columns=columns)
        prediction = model.predict(input_df)[0]
        logging.info(f"Predicted price: ${prediction:,.0f}")
        return prediction
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
