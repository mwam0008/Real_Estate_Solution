"""
utils.py - Visualization helpers for Real Estate Price Prediction App
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)


def plot_price_distribution(df: pd.DataFrame):
    """Histogram of house prices."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['price'], bins=40, color='#2196F3', edgecolor='white', alpha=0.85)
        ax.axvline(df['price'].mean(), color='red', linestyle='--', label=f"Mean: ${df['price'].mean():,.0f}")
        ax.axvline(df['price'].median(), color='orange', linestyle='--', label=f"Median: ${df['price'].median():,.0f}")
        ax.set_title('House Price Distribution', fontweight='bold')
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Price distribution plot error: {e}")
        raise


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of all features."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, linewidths=0.5, annot_kws={'size': 8})
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Heatmap error: {e}")
        raise


def plot_actual_vs_predicted(y_test, test_pred, model_name: str):
    """Scatter plot of actual vs predicted prices."""
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, test_pred, alpha=0.4, color='#2196F3', edgecolors='white', s=30)
        # Perfect prediction line
        min_val = min(y_test.min(), test_pred.min())
        max_val = max(y_test.max(), test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title(f'{model_name} — Actual vs Predicted', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Actual vs predicted plot error: {e}")
        raise


def plot_mae_comparison(lr_mae: float, rf_mae: float):
    """Bar chart comparing MAE of both models."""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['Linear Regression', 'Random Forest']
        maes = [lr_mae, rf_mae]
        colors = ['#FF9800', '#4CAF50']
        bars = ax.bar(models, maes, color=colors, edgecolor='white', linewidth=2)

        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                    f'${mae:,.0f}', ha='center', fontweight='bold')

        ax.axhline(70000, color='red', linestyle='--', label='Target MAE ($70,000)')
        ax.set_ylabel('Mean Absolute Error (USD)')
        ax.set_title('Model Comparison — MAE', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"MAE comparison plot error: {e}")
        raise


def plot_feature_importance(model, columns):
    """Horizontal bar chart of Random Forest feature importances."""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importances[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([columns[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest — Feature Importance', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Feature importance plot error: {e}")
        raise


def plot_price_by_beds(df: pd.DataFrame):
    """Box plot of price distribution by number of bedrooms."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column='price', by='beds', ax=ax)
        ax.set_title('Price by Number of Bedrooms', fontweight='bold')
        ax.set_xlabel('Bedrooms')
        ax.set_ylabel('Price (USD)')
        plt.suptitle('')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Price by beds plot error: {e}")
        raise
