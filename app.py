"""
app.py - Streamlit Web App for Real Estate Price Prediction
Models: Linear Regression + Random Forest
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from model import (
    load_data,
    split_data,
    train_linear_regression,
    train_random_forest,
    evaluate_model,
    save_model,
    load_model,
    predict_price,
    FEATURES,
)
from utils import (
    plot_price_distribution,
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_mae_comparison,
    plot_feature_importance,
    plot_price_by_beds,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Price Predictor",
    layout="wide"
)

st.title("Real Estate Price Prediction")
st.markdown("Predict house prices using **Linear Regression** and **Random Forest** models.")

# ── Load Data ─────────────────────────────────────────────────
DATA_PATH = "final.csv"

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

try:
    df = get_data()
except Exception as e:
    st.error(f"Could not load final.csv. Make sure it's in the same folder.\nError: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", [
    "Data Overview",
    "Train & Compare Models",
    "Predict House Price",
])

# ════════════════════════════════════════════════════════════
# SECTION 1 - Data Overview
# ════════════════════════════════════════════════════════════
if section == "Data Overview":
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Avg Price", f"${df['price'].mean():,.0f}")
    col4.metric("Price Range", f"${df['price'].min():,} – ${df['price'].max():,}")

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Price Distribution")
    fig = plot_price_distribution(df)
    st.pyplot(fig)

    st.subheader("🛏️ Price by Bedrooms")
    fig2 = plot_price_by_beds(df)
    st.pyplot(fig2)

    st.subheader("Feature Correlation Heatmap")
    st.markdown("Shows how strongly each feature relates to price. Brighter red = stronger positive relationship.")
    fig3 = plot_correlation_heatmap(df)
    st.pyplot(fig3)

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe().round(2))

# ════════════════════════════════════════════════════════════
# SECTION 2 - Train & Compare Models
# ════════════════════════════════════════════════════════════
elif section == "Train & Compare Models":
    st.header("Train & Compare Models")
    st.markdown("""
    Train both models on the real estate data and compare their performance.
    The goal is to get **MAE below $70,000**.
    """)

    st.sidebar.subheader("Settings")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)
    n_estimators = st.sidebar.slider("Random Forest Trees", 50, 500, 200, step=50)

    if st.button("Train Both Models"):
        with st.spinner("Training Linear Regression and Random Forest... ⏳"):
            try:
                x_train, x_test, y_train, y_test = split_data(df, test_size=test_size)

                # Train both
                lrmodel = train_linear_regression(x_train, y_train)
                rfmodel = train_random_forest(x_train, y_train, n_estimators=n_estimators)

                # Evaluate both
                lr_train_mae, lr_test_mae, lr_train_pred, lr_test_pred = evaluate_model(
                    lrmodel, x_train, y_train, x_test, y_test)
                rf_train_mae, rf_test_mae, rf_train_pred, rf_test_pred = evaluate_model(
                    rfmodel, x_train, y_train, x_test, y_test)

                # Save best model
                save_model(rfmodel, 'RE_Model.pkl')
                st.session_state['model_trained'] = True
                st.session_state['columns'] = list(x_train.columns)

                st.success("Both models trained! Random Forest saved as RE_Model.pkl")

                # Metrics
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Linear Regression**")
                    st.metric("Train MAE", f"${lr_train_mae:,.0f}")
                    st.metric("Test MAE", f"${lr_test_mae:,.0f}",
                              delta=f"{'Under target' if lr_test_mae < 70000 else 'Over $70k target'}")

                with col2:
                    st.markdown("**Random Forest**")
                    st.metric("Train MAE", f"${rf_train_mae:,.0f}")
                    st.metric("Test MAE", f"${rf_test_mae:,.0f}",
                              delta=f"{'Under target' if rf_test_mae < 70000 else 'Over $70k target'}")

                # MAE comparison chart
                st.subheader("MAE Comparison")
                fig = plot_mae_comparison(lr_test_mae, rf_test_mae)
                st.pyplot(fig)

                # Actual vs Predicted
                st.subheader("Actual vs Predicted Prices")
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = plot_actual_vs_predicted(y_test, lr_test_pred, "Linear Regression")
                    st.pyplot(fig2)
                with col2:
                    fig3 = plot_actual_vs_predicted(y_test, rf_test_pred, "Random Forest")
                    st.pyplot(fig3)

                # Feature importance
                st.subheader("Random Forest Feature Importance")
                st.markdown("Which features matter most for predicting price?")
                fig4 = plot_feature_importance(rfmodel, list(x_train.columns))
                st.pyplot(fig4)

                # Winner
                winner = "Random Forest" if rf_test_mae < lr_test_mae else "Linear Regression"
                st.info(f"**{winner}** performs better with lower Test MAE!")

            except Exception as e:
                st.error(f"Training failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 3 - Predict House Price
# ════════════════════════════════════════════════════════════
elif section == "Predict House Price":
    st.header("Predict a House Price")
    st.markdown("Fill in the property details and get a predicted price.")

    # Try loading saved model
    try:
        model = load_model('RE_Model.pkl')
        x_train, x_test, y_train, y_test = split_data(df)
        columns = list(x_train.columns)
        model_available = True
    except Exception:
        model_available = False

    if not model_available:
        st.warning("No trained model found. Please go to **Train & Compare Models** first!")
    else:
        st.success("Random Forest model loaded!")

        col1, col2 = st.columns(2)

        with col1:
            year_sold    = st.slider("Year Sold", 2000, 2025, 2013)
            property_tax = st.number_input("Property Tax ($/yr)", min_value=0, max_value=5000, value=216)
            insurance    = st.number_input("Insurance ($/yr)", min_value=0, max_value=2000, value=74)
            beds         = st.slider("Bedrooms", 1, 8, 3)
            baths        = st.slider("Bathrooms", 1, 6, 2)
            sqft         = st.number_input("Square Footage", min_value=200, max_value=10000, value=1500)
            year_built   = st.slider("Year Built", 1900, 2024, 1990)

        with col2:
            lot_size     = st.number_input("Lot Size (sq ft)", min_value=0, max_value=100000, value=5000)
            basement     = st.selectbox("Basement?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            popular      = st.selectbox("Popular Area?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            recession    = st.selectbox("Sold During Recession?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            property_age = st.number_input("Property Age (years)", min_value=0, max_value=150,
                                           value=year_sold - year_built)
            prop_type    = st.selectbox("Property Type", [0, 1],
                                        format_func=lambda x: "Condo" if x else "House")

        if st.button("🔮 Predict Price"):
            try:
                input_dict = {
                    'year_sold': year_sold,
                    'property_tax': property_tax,
                    'insurance': insurance,
                    'beds': beds,
                    'baths': baths,
                    'sqft': sqft,
                    'year_built': year_built,
                    'lot_size': lot_size,
                    'basement': basement,
                    'popular': popular,
                    'recession': recession,
                    'property_age': property_age,
                    'property_type_Condo': prop_type,
                }

                prediction = predict_price(model, input_dict, columns)

                st.success(f"### Predicted Price: **${prediction:,.0f}**")

                # Context
                avg = df['price'].mean()
                diff = prediction - avg
                direction = "above" if diff > 0 else "below"
                st.caption(f"This is ${abs(diff):,.0f} {direction} the dataset average of ${avg:,.0f}")

                # Show input summary
                with st.expander("Your Input Summary"):
                    st.dataframe(pd.DataFrame([input_dict]).T.rename(columns={0: 'Value'}))

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Project**")
st.sidebar.markdown("Real Estate Price Prediction")
