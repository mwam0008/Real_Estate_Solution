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
    load_data, split_data,
    train_linear_regression, train_random_forest,
    evaluate_model, save_model, load_model,
    predict_price, FEATURES,
)
from utils import (
    plot_price_distribution, plot_correlation_heatmap,
    plot_actual_vs_predicted, plot_mae_comparison,
    plot_feature_importance, plot_price_by_beds,
)
from logger import (
    log_app_start, log_page_visit, log_data_loaded,
    log_model_training, log_model_results, log_model_saved,
    log_model_loaded, log_prediction, log_error, log_warning,
    get_log_contents, get_log_line_count,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")

# Log app startup once per session
if "app_started" not in st.session_state:
    log_app_start()
    st.session_state["app_started"] = True

st.title("Real Estate Price Prediction")
st.markdown("Predict house prices using **Linear Regression** and **Random Forest** models.")

# ── Load Data ─────────────────────────────────────────────────
DATA_PATH = "final.csv"

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    log_data_loaded(DATA_PATH, df.shape[0], df.shape[1])
    return df

try:
    df = get_data()
except Exception as e:
    log_error("Data loading", e)
    st.error(f"Could not load final.csv. Make sure it's in the same folder.\nError: {e}")
    st.stop()

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", [
    "Data Overview",
    "Train & Compare Models",
    "Predict House Price",
    "Activity Log",
])

# Log every page visit (only when it changes)
if st.session_state.get("current_section") != section:
    log_page_visit(section)
    st.session_state["current_section"] = section


# ════════════════════════════════════════════════════════════
# SECTION 1 - Data Overview
# ════════════════════════════════════════════════════════════
if section == "Data Overview":
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", df.shape[0])
    col2.metric("Features",         df.shape[1] - 1)
    col3.metric("Avg Price",        f"${df['price'].mean():,.0f}")
    col4.metric("Price Range",      f"${df['price'].min():,} – ${df['price'].max():,}")

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Price Distribution")
    try:
        st.pyplot(plot_price_distribution(df))
    except Exception as e:
        log_error("plot_price_distribution", e)
        st.error("Could not render price distribution chart.")

    st.subheader("Price by Bedrooms")
    try:
        st.pyplot(plot_price_by_beds(df))
    except Exception as e:
        log_error("plot_price_by_beds", e)
        st.error("Could not render bedrooms chart.")

    st.subheader("Feature Correlation Heatmap")
    st.markdown("Shows how strongly each feature relates to price.")
    try:
        st.pyplot(plot_correlation_heatmap(df))
    except Exception as e:
        log_error("plot_correlation_heatmap", e)
        st.error("Could not render heatmap.")

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
    test_size    = st.sidebar.slider("Test Set Size",        0.1, 0.4, 0.2, step=0.05)
    n_estimators = st.sidebar.slider("Random Forest Trees", 50, 500, 200, step=50)

    if st.button("Train Both Models"):
        with st.spinner("Training Linear Regression and Random Forest..."):
            try:
                x_train, x_test, y_train, y_test = split_data(df, test_size=test_size)

                # Linear Regression
                log_model_training("Linear Regression", test_size=test_size)
                lrmodel = train_linear_regression(x_train, y_train)
                lr_train_mae, lr_test_mae, lr_train_pred, lr_test_pred = evaluate_model(
                    lrmodel, x_train, y_train, x_test, y_test)
                log_model_results("Linear Regression", lr_train_mae, lr_test_mae)

                # Random Forest
                log_model_training("Random Forest", n_estimators=n_estimators, test_size=test_size)
                rfmodel = train_random_forest(x_train, y_train, n_estimators=n_estimators)
                rf_train_mae, rf_test_mae, rf_train_pred, rf_test_pred = evaluate_model(
                    rfmodel, x_train, y_train, x_test, y_test)
                log_model_results("Random Forest", rf_train_mae, rf_test_mae)

                # Save best model
                save_model(rfmodel, "RE_Model.pkl")
                log_model_saved("RE_Model.pkl")

                st.session_state["model_trained"] = True
                st.session_state["columns"]       = list(x_train.columns)

                st.success("Both models trained! Random Forest saved as RE_Model.pkl")

                # ── Metrics ─────────────────────────────────
                st.subheader("Model Performance")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Linear Regression**")
                    st.metric("Train MAE", f"${lr_train_mae:,.0f}")
                    st.metric("Test MAE",  f"${lr_test_mae:,.0f}",
                              delta="Under target" if lr_test_mae < 70000 else "Over $70k target")
                with c2:
                    st.markdown("**Random Forest**")
                    st.metric("Train MAE", f"${rf_train_mae:,.0f}")
                    st.metric("Test MAE",  f"${rf_test_mae:,.0f}",
                              delta="Under target" if rf_test_mae < 70000 else "Over $70k target")

                # ── Charts ──────────────────────────────────
                st.subheader("MAE Comparison")
                st.pyplot(plot_mae_comparison(lr_test_mae, rf_test_mae))

                st.subheader("Actual vs Predicted Prices")
                c3, c4 = st.columns(2)
                with c3:
                    st.pyplot(plot_actual_vs_predicted(y_test, lr_test_pred, "Linear Regression"))
                with c4:
                    st.pyplot(plot_actual_vs_predicted(y_test, rf_test_pred, "Random Forest"))

                st.subheader("Random Forest Feature Importance")
                st.pyplot(plot_feature_importance(rfmodel, list(x_train.columns)))

                winner = "Random Forest" if rf_test_mae < lr_test_mae else "Linear Regression"
                st.info(f"**{winner}** performs better with lower Test MAE!")

            except Exception as e:
                log_error("Train & Compare Models", e)
                st.error(f"Training failed: {e}")


# ════════════════════════════════════════════════════════════
# SECTION 3 - Predict House Price
# ════════════════════════════════════════════════════════════
elif section == "Predict House Price":
    st.header("Predict a House Price")
    st.markdown("Fill in the property details and get a predicted price.")

    try:
        model = load_model("RE_Model.pkl")
        log_model_loaded("RE_Model.pkl")
        x_train, x_test, y_train, y_test = split_data(df)
        columns = list(x_train.columns)
        model_available = True
    except Exception:
        model_available = False
        log_warning("No trained model found — user directed to train first.")

    if not model_available:
        st.warning("No trained model found. Please go to **Train & Compare Models** first!")
    else:
        st.success("Random Forest model loaded!")

        c1, c2 = st.columns(2)
        with c1:
            year_sold    = st.slider("Year Sold",         2000, 2025, 2013)
            property_tax = st.number_input("Property Tax ($/yr)", 0, 5000, 216)
            insurance    = st.number_input("Insurance ($/yr)",    0, 2000,  74)
            beds         = st.slider("Bedrooms",  1, 8, 3)
            baths        = st.slider("Bathrooms", 1, 6, 2)
            sqft         = st.number_input("Square Footage", 200, 10000, 1500)
            year_built   = st.slider("Year Built", 1900, 2024, 1990)
        with c2:
            lot_size     = st.number_input("Lot Size (sq ft)", 0, 100000, 5000)
            basement     = st.selectbox("Basement?",               [0, 1], format_func=lambda x: "Yes" if x else "No")
            popular      = st.selectbox("Popular Area?",           [0, 1], format_func=lambda x: "Yes" if x else "No")
            recession    = st.selectbox("Sold During Recession?",  [0, 1], format_func=lambda x: "Yes" if x else "No")
            property_age = st.number_input("Property Age (years)", 0, 150, year_sold - year_built)
            prop_type    = st.selectbox("Property Type",           [0, 1], format_func=lambda x: "Condo" if x else "House")

        if st.button("Predict Price"):
            input_dict = {
                "year_sold": year_sold, "property_tax": property_tax,
                "insurance": insurance, "beds": beds, "baths": baths,
                "sqft": sqft, "year_built": year_built, "lot_size": lot_size,
                "basement": basement, "popular": popular, "recession": recession,
                "property_age": property_age, "property_type_Condo": prop_type,
            }
            try:
                prediction = predict_price(model, input_dict, columns)
                log_prediction(input_dict, prediction)

                st.success(f"### Predicted Price: **${prediction:,.0f}**")

                avg  = df["price"].mean()
                diff = prediction - avg
                direction = "above" if diff > 0 else "below"
                st.caption(f"This is ${abs(diff):,.0f} {direction} the dataset average of ${avg:,.0f}")

                with st.expander("Your Input Summary"):
                    st.dataframe(pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}))

            except Exception as e:
                log_error("Predict House Price", e)
                st.error(f"Prediction failed: {e}")


# ════════════════════════════════════════════════════════════
# SECTION 4 - Activity Log
# ════════════════════════════════════════════════════════════
elif section == "Activity Log":
    st.header("Activity Log")
    st.markdown(
        "All app events — data loads, model training, predictions, and errors — "
        "are recorded in `app_activity.txt`."
    )

    log_text = get_log_contents()
    line_count = get_log_line_count()

    col1, col2 = st.columns(2)
    col1.metric("Total Log Lines", line_count)
    col2.metric("Log File", "app_activity.txt")

    st.subheader("Log Contents")
    st.text_area("app_activity.txt", value=log_text, height=500)

    # Download button
    st.download_button(
        label="Download app_activity.txt",
        data=log_text,
        file_name="app_activity.txt",
        mime="text/plain",
    )

    if st.button("Clear Log"):
        try:
            with open("app_activity.txt", "w") as f:
                f.write("")
            log_app_start()
            st.success("Log cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not clear log: {e}")


# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Project**")
st.sidebar.markdown("Real Estate Price Prediction")
st.sidebar.markdown(f"Log lines: {get_log_line_count()}")
