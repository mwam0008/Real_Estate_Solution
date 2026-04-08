# Real Estate Price Prediction

A Streamlit web app that predicts house prices using **Linear Regression** and **Random Forest** models.

## What This App Does

| Section | What it shows |
|---|---|
| Data Overview | Price distribution, bedroom analysis, correlation heatmap |
| Train & Compare Models | Train both models, compare MAE, actual vs predicted charts, feature importance |
| Predict House Price | Enter property details → get predicted price |

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/real-estate-predictor.git
cd real-estate-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
real_estate_app/
├── app.py            ← Streamlit web app
├── model.py          ← Linear Regression + Random Forest logic
├── utils.py          ← All charts and visualizations
├── final.csv         ← Real estate dataset (1860 properties)
├── requirements.txt  ← Dependencies
└── README.md         ← This file
```

## Models Used

| Model | Train MAE | Notes |
|---|---|---|
| Linear Regression | Higher | Simple baseline |
| Random Forest | Lower | Better, saved as RE_Model.pkl |

**Target:** MAE below $70,000

## Key Concepts

- **Train/Test Split** — 80% training, 20% testing with stratification
- **Mean Absolute Error (MAE)** — average dollar difference between predicted and actual price
- **Random Forest** — ensemble of decision trees, more accurate than linear regression
- **Pickle** — saves the trained model to disk for reuse
- **Feature Importance** — shows which features matter most for price prediction

## Dataset Features

`year_sold`, `property_tax`, `insurance`, `beds`, `baths`, `sqft`, `year_built`, `lot_size`, `basement`, `popular`, `recession`, `property_age`, `property_type_Condo`
