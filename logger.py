"""
logger.py - File-based activity logger for Real Estate Price Prediction App
Writes all app events to app_activity.txt
"""

import logging
import os
from datetime import datetime

LOG_FILE = "app_activity.txt"

# ── Configure file logger ─────────────────────────────────────
def _get_logger() -> logging.Logger:
    """Return a configured logger that writes to app_activity.txt."""
    logger = logging.getLogger("real_estate_app")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


_logger = _get_logger()


# ── Public logging functions ──────────────────────────────────

def log_app_start() -> None:
    """Log when the Streamlit app starts up."""
    _logger.info("=" * 60)
    _logger.info("APP STARTED — Real Estate Price Predictor")
    _logger.info("=" * 60)


def log_page_visit(page: str) -> None:
    """Log which section/page the user navigated to.

    Args:
        page: Name of the page or section visited.
    """
    _logger.info(f"PAGE VISIT        | section='{page}'")


def log_data_loaded(filepath: str, rows: int, cols: int) -> None:
    """Log successful data load.

    Args:
        filepath: Path to the CSV file loaded.
        rows: Number of rows in the dataset.
        cols: Number of columns in the dataset.
    """
    _logger.info(f"DATA LOADED       | file='{filepath}' rows={rows} cols={cols}")


def log_model_training(model_name: str, n_estimators: int = None,
                        test_size: float = None) -> None:
    """Log when a model training run starts.

    Args:
        model_name: Name of the model being trained.
        n_estimators: Number of trees (Random Forest only).
        test_size: Fraction used for test split.
    """
    extras = ""
    if n_estimators:
        extras += f" n_estimators={n_estimators}"
    if test_size:
        extras += f" test_size={test_size}"
    _logger.info(f"TRAINING STARTED  | model='{model_name}'{extras}")


def log_model_results(model_name: str, train_mae: float, test_mae: float,
                       target: float = 70000.0) -> None:
    """Log model evaluation results.

    Args:
        model_name: Name of the evaluated model.
        train_mae: Training set MAE in dollars.
        test_mae: Test set MAE in dollars.
        target: MAE goal threshold in dollars.
    """
    goal = "MET" if test_mae < target else "NOT MET"
    _logger.info(
        f"MODEL RESULTS     | model='{model_name}' "
        f"train_mae=${train_mae:,.0f} test_mae=${test_mae:,.0f} "
        f"goal_${target:,.0f}={goal}"
    )


def log_model_saved(path: str) -> None:
    """Log when a model is saved to disk.

    Args:
        path: File path where the model was saved.
    """
    _logger.info(f"MODEL SAVED       | path='{path}'")


def log_model_loaded(path: str) -> None:
    """Log when a model is loaded from disk.

    Args:
        path: File path from which the model was loaded.
    """
    _logger.info(f"MODEL LOADED      | path='{path}'")


def log_prediction(input_summary: dict, predicted_price: float) -> None:
    """Log a price prediction request and result.

    Args:
        input_summary: Key property features used for prediction.
        predicted_price: The predicted house price in dollars.
    """
    beds    = input_summary.get("beds", "?")
    baths   = input_summary.get("baths", "?")
    sqft    = input_summary.get("sqft", "?")
    year    = input_summary.get("year_sold", "?")
    ptype   = "Condo" if input_summary.get("property_type_Condo") else "House"
    _logger.info(
        f"PREDICTION        | "
        f"beds={beds} baths={baths} sqft={sqft} year_sold={year} "
        f"type={ptype} predicted=${predicted_price:,.0f}"
    )


def log_error(context: str, error: Exception) -> None:
    """Log an application error.

    Args:
        context: Description of where the error occurred.
        error: The exception that was raised.
    """
    _logger.error(f"ERROR             | context='{context}' error={type(error).__name__}: {error}")


def log_warning(message: str) -> None:
    """Log a non-critical warning.

    Args:
        message: Warning message to record.
    """
    _logger.warning(f"WARNING           | {message}")


def get_log_contents() -> str:
    """Read and return the full contents of the log file.

    Returns:
        Log file contents as a string, or a message if not found.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No log file found yet. Activity will appear here after the app is used."
    except Exception as e:
        return f"Could not read log file: {e}"


def get_log_line_count() -> int:
    """Return the number of lines currently in the log file.

    Returns:
        Number of lines in the log file, or 0 if not found.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
