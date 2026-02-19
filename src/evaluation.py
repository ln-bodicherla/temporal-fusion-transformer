"""
TFT evaluation and reporting.

Computes per-horizon metrics, generates prediction plots with confidence
intervals, checks calibration of quantile forecasts, and produces
comprehensive evaluation reports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TFTEvaluator:
    """Evaluation suite for Temporal Fusion Transformer predictions."""

    def __init__(
        self,
        prediction_length: int = 20,
        quantiles: Optional[List[float]] = None,
    ):
        self.prediction_length = prediction_length
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    def compute_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute RMSE, MAE, MAPE, SMAPE per prediction horizon.

        Args:
            actuals: Array of shape (n_samples, prediction_length)
            predictions: Array of shape (n_samples, prediction_length) or
                         (n_samples, prediction_length, n_quantiles)
        """
        if predictions.ndim == 3:
            median_idx = len(self.quantiles) // 2
            point_predictions = predictions[:, :, median_idx]
        else:
            point_predictions = predictions

        overall_metrics = self._compute_point_metrics(actuals, point_predictions)

        per_horizon = {}
        for h in range(min(actuals.shape[1], point_predictions.shape[1])):
            horizon_actuals = actuals[:, h]
            horizon_preds = point_predictions[:, h]
            per_horizon[f"horizon_{h + 1}"] = self._compute_point_metrics(
                horizon_actuals.reshape(-1),
                horizon_preds.reshape(-1),
            )

        result = {
            "overall": overall_metrics,
            "per_horizon": per_horizon,
            "num_samples": int(actuals.shape[0]),
            "prediction_length": int(actuals.shape[1]),
        }

        if predictions.ndim == 3:
            calibration = self.compute_calibration(actuals, predictions)
            result["calibration"] = calibration

        return result

    def _compute_point_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard regression metrics between actuals and predictions."""
        errors = actuals.flatten() - predictions.flatten()
        abs_errors = np.abs(errors)
        actuals_flat = actuals.flatten()
        preds_flat = predictions.flatten()

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(abs_errors))

        denominator = np.abs(actuals_flat) + 1e-8
        mape = float(np.mean(abs_errors / denominator) * 100)

        smape_denom = (np.abs(actuals_flat) + np.abs(preds_flat)) / 2 + 1e-8
        smape = float(np.mean(abs_errors / smape_denom) * 100)

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals_flat - np.mean(actuals_flat)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "smape": round(smape, 4),
            "r2": round(r2, 4),
        }

    def plot_predictions(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot actual vs predicted for multiple horizons and samples.

        Shows a grid of individual prediction examples with full
        encoder context and decoder predictions overlaid.
        """
        if predictions.ndim == 3:
            median_idx = len(self.quantiles) // 2
            point_preds = predictions[:, :, median_idx]
        else:
            point_preds = predictions

        if sample_indices is None:
            n_samples = min(6, actuals.shape[0])
            sample_indices = np.linspace(
                0, actuals.shape[0] - 1, n_samples, dtype=int
            ).tolist()

        n_plots = len(sample_indices)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for plot_idx, sample_idx in enumerate(sample_indices):
            ax = axes[plot_idx]
            horizons = np.arange(1, actuals.shape[1] + 1)

            ax.plot(
                horizons, actuals[sample_idx], "o-",
                color="#1976D2", label="Actual", markersize=4,
            )
            ax.plot(
                horizons, point_preds[sample_idx], "s--",
                color="#E53935", label="Predicted", markersize=4,
            )

            ax.set_xlabel("Prediction Horizon")
            ax.set_ylabel("RUL")
            ax.set_title(f"Sample {sample_idx}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Multi-Horizon Predictions", fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Prediction plot saved to {save_path}")
        return fig

    def plot_prediction_intervals(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot prediction intervals (confidence bands) for a single sample.

        Requires predictions to include quantile estimates
        (shape: n_samples x prediction_length x n_quantiles).
        """
        if predictions.ndim != 3:
            raise ValueError(
                "Quantile predictions required (3D array). "
                "Got shape: {}".format(predictions.shape)
            )

        fig, ax = plt.subplots(figsize=(12, 6))
        horizons = np.arange(1, actuals.shape[1] + 1)

        q_labels = {
            (0, -1): ("90% PI", 0.15),
            (1, -2): ("50% PI", 0.25),
        }

        colors = ["#BBDEFB", "#64B5F6"]
        for (lo_idx, hi_idx), (label, alpha) in q_labels.items():
            lo = predictions[sample_idx, :, lo_idx]
            hi = predictions[sample_idx, :, hi_idx]
            ax.fill_between(
                horizons, lo, hi,
                alpha=alpha + 0.2,
                color=colors.pop(0),
                label=label,
            )

        median_idx = len(self.quantiles) // 2
        ax.plot(
            horizons, predictions[sample_idx, :, median_idx],
            "-", color="#1565C0", linewidth=2, label="Median Prediction",
        )
        ax.plot(
            horizons, actuals[sample_idx],
            "ko-", markersize=5, label="Actual",
        )

        ax.set_xlabel("Prediction Horizon", fontsize=12)
        ax.set_ylabel("RUL", fontsize=12)
        ax.set_title(
            f"Prediction Intervals - Sample {sample_idx}", fontsize=14
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Interval plot saved to {save_path}")
        return fig

    def compute_calibration(
        self,
        actuals: np.ndarray,
        quantile_predictions: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate prediction interval calibration.

        For each quantile, computes the empirical coverage (fraction of
        actuals below the quantile prediction). A well-calibrated model
        should have empirical coverage close to the nominal quantile level.
        """
        if quantile_predictions.ndim != 3:
            raise ValueError("Requires quantile predictions (3D array)")

        calibration = {}
        for q_idx, q_level in enumerate(self.quantiles):
            below = actuals < quantile_predictions[:, :, q_idx]
            empirical_coverage = float(np.mean(below))
            calibration[f"q{q_level:.2f}"] = {
                "nominal": q_level,
                "empirical": round(empirical_coverage, 4),
                "deviation": round(empirical_coverage - q_level, 4),
            }

        interval_pairs = [(0, -1), (1, -2)]
        interval_labels = ["90%", "50%"]
        for (lo_idx, hi_idx), label in zip(interval_pairs, interval_labels):
            lo = quantile_predictions[:, :, lo_idx]
            hi = quantile_predictions[:, :, hi_idx]
            within = (actuals >= lo) & (actuals <= hi)
            nominal = self.quantiles[hi_idx] - self.quantiles[lo_idx]
            calibration[f"interval_{label}"] = {
                "nominal_coverage": round(nominal, 4),
                "empirical_coverage": round(float(np.mean(within)), 4),
            }

        return calibration

    def generate_report(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        baseline_metrics: Optional[Dict[str, float]] = None,
        output_dir: str = "reports",
    ) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report with plots and metrics.

        Produces:
        - metrics.json: All computed metrics
        - predictions.png: Actual vs predicted plots
        - intervals.png: Prediction interval visualization (if quantiles available)
        - calibration.json: Calibration analysis (if quantiles available)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_metrics(actuals, predictions)

        if baseline_metrics:
            metrics["baseline"] = baseline_metrics
            if "naive_rmse" in baseline_metrics:
                skill = 1.0 - (
                    metrics["overall"]["rmse"] / baseline_metrics["naive_rmse"]
                )
                metrics["overall"]["forecast_skill"] = round(skill, 4)

        self.plot_predictions(
            actuals, predictions,
            save_path=str(output_path / "predictions.png"),
        )

        if predictions.ndim == 3:
            self.plot_prediction_intervals(
                actuals, predictions,
                save_path=str(output_path / "intervals.png"),
            )

        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {output_dir}/")
        logger.info(
            f"Overall RMSE: {metrics['overall']['rmse']}, "
            f"MAE: {metrics['overall']['mae']}"
        )

        return metrics
