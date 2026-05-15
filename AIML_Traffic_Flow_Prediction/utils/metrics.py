"""
Evaluation metrics for traffic flow prediction.
All metrics operate on numpy arrays after inverse-transforming predictions.
"""
import numpy as np


def mask_mape(preds, labels, null_val=0.0):
    """MAPE ignoring zero/null ground-truth values."""
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mape = np.abs((preds - labels) / (labels + 1e-8))
    mape = np.nan_to_num(mask * mape)
    return float(np.mean(mape)) * 100.0  # return as %


def mae(preds, labels):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(preds - labels)))


def rmse(preds, labels):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def mape(preds, labels, eps=1e-8):
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((preds - labels) / (np.abs(labels) + eps)))) * 100.0


def r2_score(preds, labels):
    """Coefficient of Determination R²."""
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def compute_all_metrics(preds, labels, null_val=0.0):
    """
    Compute all evaluation metrics.

    Args:
        preds  : numpy array, model predictions (denormalized)
        labels : numpy array, ground truth (denormalized)
        null_val: value to mask for MAPE (default 0.0)

    Returns:
        dict with keys: MAE, RMSE, MAPE, R2
    """
    return {
        "MAE":  mae(preds, labels),
        "RMSE": rmse(preds, labels),
        "MAPE": mask_mape(preds, labels, null_val),
        "R2":   r2_score(preds, labels),
    }


def print_metrics(metrics_dict, model_name="Model"):
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*45}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*45}")
    print(f"  MAE  : {metrics_dict['MAE']:.4f} km/h")
    print(f"  RMSE : {metrics_dict['RMSE']:.4f} km/h")
    print(f"  MAPE : {metrics_dict['MAPE']:.2f} %")
    print(f"  R²   : {metrics_dict['R2']:.4f}")
    print(f"{'='*45}\n")
