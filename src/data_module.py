"""
TimeSeriesDataModule for Temporal Fusion Transformer.

Handles data loading, preprocessing, synthetic data generation, and conversion
to pytorch-forecasting TimeSeriesDataSet format with proper train/val/test splits.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TimeSeriesDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-variate sensor time series."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config["data"]
        self.training_config = self.config["training"]
        self.synthetic_config = self.config.get("synthetic_data", {})

        self.df: Optional[pd.DataFrame] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None
        self.validation_dataset: Optional[TimeSeriesDataSet] = None
        self.test_dataset: Optional[TimeSeriesDataSet] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}

    def prepare_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load and preprocess sensor time series data.

        If no data path is provided, generates synthetic degradation data
        suitable for RUL prediction tasks.
        """
        if data_path is not None:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            df = self._preprocess_raw_data(df)
        else:
            logger.info("No data path provided, generating synthetic sensor data")
            df = self.generate_synthetic_sensor_data()

        df = self._validate_columns(df)
        df = self._encode_categoricals(df)
        df = self._clip_rul(df, max_rul=125)

        self.df = df
        logger.info(
            f"Prepared dataset: {len(df)} rows, "
            f"{df[self.data_config['group_ids'][0]].nunique()} units"
        )
        return df

    def _preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format raw CSV data to expected schema."""
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        if "rul" not in df.columns:
            if "cycle" in df.columns and "unit_id" in df.columns:
                max_cycles = df.groupby("unit_id")["cycle"].transform("max")
                df["rul"] = max_cycles - df["cycle"]

        df = df.sort_values(
            [self.data_config["group_ids"][0], self.data_config["time_idx"]]
        ).reset_index(drop=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(0)

        return df

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist in the dataframe."""
        required = (
            [self.data_config["target"], self.data_config["time_idx"]]
            + self.data_config["group_ids"]
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode categorical columns for pytorch-forecasting."""
        categoricals = self.data_config.get("static_categoricals", [])
        for col in categoricals:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
        for gid in self.data_config["group_ids"]:
            df[gid] = df[gid].astype(str)
        return df

    @staticmethod
    def _clip_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
        """Clip RUL values to a maximum threshold (piece-wise linear)."""
        df["rul"] = df["rul"].clip(upper=max_rul).astype(float)
        return df

    def generate_synthetic_sensor_data(self) -> pd.DataFrame:
        """Generate multi-variate degradation series for demo purposes.

        Creates realistic sensor readings from simulated turbofan engines
        with progressive degradation patterns and varying failure modes.
        """
        num_units = self.synthetic_config.get("num_units", 100)
        max_cycles = self.synthetic_config.get("max_cycles", 300)
        num_sensors = self.synthetic_config.get("num_sensors", 7)
        noise_std = self.synthetic_config.get("noise_std", 0.05)
        seed = self.synthetic_config.get("seed", 42)

        rng = np.random.RandomState(seed)
        records: List[Dict[str, Any]] = []

        sensor_baselines = rng.uniform(20.0, 100.0, size=num_sensors)
        degradation_rates = rng.uniform(0.01, 0.08, size=num_sensors)

        for unit_id in range(1, num_units + 1):
            lifetime = rng.randint(int(max_cycles * 0.4), max_cycles + 1)
            op_setting = rng.choice(["regime_1", "regime_2", "regime_3"])

            unit_offset = rng.normal(0, 2.0, size=num_sensors)
            failure_mode = rng.choice(["thermal", "mechanical", "vibration"])

            mode_multipliers = np.ones(num_sensors)
            if failure_mode == "thermal":
                mode_multipliers[:3] *= rng.uniform(1.5, 2.5)
            elif failure_mode == "mechanical":
                mode_multipliers[2:5] *= rng.uniform(1.5, 2.5)
            else:
                mode_multipliers[4:] *= rng.uniform(1.5, 2.5)

            for cycle in range(1, lifetime + 1):
                progress = cycle / lifetime
                degradation = (
                    degradation_rates * mode_multipliers * (progress ** 1.5)
                )
                noise = rng.normal(0, noise_std, size=num_sensors)

                sensor_values = (
                    sensor_baselines + unit_offset + degradation * cycle + noise
                )

                record = {
                    "unit_id": str(unit_id),
                    "cycle": cycle,
                    "op_setting": op_setting,
                    "rul": float(lifetime - cycle),
                }
                for s_idx in range(num_sensors):
                    record[f"sensor_{s_idx + 1}"] = sensor_values[s_idx]
                records.append(record)

        df = pd.DataFrame(records)
        logger.info(
            f"Generated synthetic data: {num_units} units, "
            f"{len(df)} total rows, {num_sensors} sensors"
        )
        return df

    def create_time_series_dataset(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> TimeSeriesDataSet:
        """Convert a pandas DataFrame to pytorch-forecasting TimeSeriesDataSet."""
        time_varying_unknown = self.data_config.get(
            "time_varying_unknown_reals", []
        )
        available_unknown = [c for c in time_varying_unknown if c in df.columns]

        kwargs = dict(
            data=df,
            time_idx=self.data_config["time_idx"],
            target=self.data_config["target"],
            group_ids=self.data_config["group_ids"],
            max_encoder_length=self.data_config["max_encoder_length"],
            max_prediction_length=self.data_config["max_prediction_length"],
            static_categoricals=self.data_config.get("static_categoricals", []),
            time_varying_known_reals=[self.data_config["time_idx"]],
            time_varying_unknown_reals=available_unknown
            + [self.data_config["target"]],
            target_normalizer=GroupNormalizer(
                groups=self.data_config["group_ids"],
                transformation="softplus",
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        if is_training:
            return TimeSeriesDataSet(**kwargs)
        else:
            return TimeSeriesDataSet(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test splits and build TimeSeriesDataSets."""
        if self.df is None:
            self.prepare_data()

        df = self.df
        unit_col = self.data_config["group_ids"][0]
        unique_units = df[unit_col].unique()
        n_units = len(unique_units)

        rng = np.random.RandomState(self.synthetic_config.get("seed", 42))
        shuffled = rng.permutation(unique_units)

        train_end = int(0.7 * n_units)
        val_end = int(0.85 * n_units)

        train_units = set(shuffled[:train_end])
        val_units = set(shuffled[train_end:val_end])
        test_units = set(shuffled[val_end:])

        train_df = df[df[unit_col].isin(train_units)].reset_index(drop=True)
        val_df = df[df[unit_col].isin(val_units)].reset_index(drop=True)
        test_df = df[df[unit_col].isin(test_units)].reset_index(drop=True)

        self.training_dataset = self.create_time_series_dataset(
            train_df, is_training=True
        )
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, val_df, stop_randomization=True
        )
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, test_df, stop_randomization=True
        )

        logger.info(
            f"Data splits - Train: {len(train_units)} units, "
            f"Val: {len(val_units)} units, Test: {len(test_units)} units"
        )

    def train_dataloader(self) -> DataLoader:
        return self.training_dataset.to_dataloader(
            train=True,
            batch_size=self.training_config["batch_size"],
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self.validation_dataset.to_dataloader(
            train=False,
            batch_size=self.training_config["batch_size"] * 2,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return self.test_dataset.to_dataloader(
            train=False,
            batch_size=self.training_config["batch_size"] * 2,
            num_workers=4,
            persistent_workers=True,
        )

    def compute_baseline(self) -> Dict[str, float]:
        """Compute naive forecast baseline (last value carried forward).

        Returns RMSE and MAE of the naive predictor for comparison.
        """
        if self.df is None:
            self.prepare_data()

        df = self.df
        target = self.data_config["target"]
        group_col = self.data_config["group_ids"][0]
        pred_len = self.data_config["max_prediction_length"]

        errors = []
        for _, group_df in df.groupby(group_col):
            group_df = group_df.sort_values(self.data_config["time_idx"])
            values = group_df[target].values
            if len(values) <= pred_len:
                continue
            last_known = values[-(pred_len + 1)]
            actuals = values[-pred_len:]
            naive_preds = np.full(pred_len, last_known)
            errors.append(actuals - naive_preds)

        all_errors = np.concatenate(errors)
        rmse = float(np.sqrt(np.mean(all_errors ** 2)))
        mae = float(np.mean(np.abs(all_errors)))

        baseline = {"naive_rmse": rmse, "naive_mae": mae}
        logger.info(f"Baseline metrics: RMSE={rmse:.3f}, MAE={mae:.3f}")
        return baseline

    def get_data_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for normalization and reporting."""
        if self.df is None:
            self.prepare_data()

        df = self.df
        target = self.data_config["target"]
        group_col = self.data_config["group_ids"][0]

        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        sensor_stats = {}
        for col in sensor_cols:
            sensor_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }

        unit_lengths = df.groupby(group_col)[self.data_config["time_idx"]].max()

        stats = {
            "num_rows": len(df),
            "num_units": df[group_col].nunique(),
            "target_stats": {
                "mean": float(df[target].mean()),
                "std": float(df[target].std()),
                "min": float(df[target].min()),
                "max": float(df[target].max()),
            },
            "sensor_stats": sensor_stats,
            "unit_lifetime_stats": {
                "mean": float(unit_lengths.mean()),
                "std": float(unit_lengths.std()),
                "min": int(unit_lengths.min()),
                "max": int(unit_lengths.max()),
            },
            "num_sensors": len(sensor_cols),
        }
        return stats
