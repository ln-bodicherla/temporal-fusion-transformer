# Temporal Fusion Transformer for Time Series Forecasting

Attention-based architecture for multi-horizon time series prediction with built-in interpretability. Provides variable importance rankings and temporal attention patterns for understanding which features and time steps drive predictions.

This project implements the Temporal Fusion Transformer (TFT) for predictive maintenance remaining useful life (RUL) estimation. Unlike single-step LSTM approaches, TFT produces quantile (probabilistic) forecasts across multiple future horizons simultaneously, with native support for static covariates, known future inputs, and observed past inputs.

Uses generated synthetic sensor-degradation data for demonstration when no `--data` CSV is supplied.

## Architecture

The TFT architecture combines several components:

- **Gated Residual Networks (GRN)** for flexible nonlinear processing
- **Variable Selection Networks** that identify relevant input variables at each time step
- **Static covariate encoders** that integrate time-invariant features
- **Temporal self-attention** over a multi-head attention mechanism for long-range dependencies
- **Quantile regression outputs** for probabilistic forecasting with calibrated prediction intervals

## Key Features

- **Multi-Horizon Forecasting**: Predict RUL at multiple future time steps in a single forward pass
- **Interpretability**: Built-in attention weights reveal temporal patterns and variable importance
- **Quantile Loss**: Prediction intervals via quantile regression (10th, 25th, 50th, 75th, 90th percentiles)
- **Variable Selection**: Automatic identification of the most informative sensors and operating conditions
- **Static Covariates**: Incorporate unit-level metadata (operating regime, fleet ID) alongside time-varying sensors

## Alternative Approach Note

This is an alternative to the LSTM-based predictive maintenance approach. TFT provides richer interpretability and multi-horizon capabilities, but LSTM was chosen for the primary project due to simpler deployment requirements and sufficient accuracy on the single-step RUL prediction task. TFT's additional complexity is most beneficial when multi-horizon forecasts and feature attribution are critical requirements.

## Project Structure

```
temporal-fusion-transformer/
├── src/
│   ├── data_module.py        # Data loading and preprocessing
│   ├── tft_model.py          # TFT model training and interpretation
│   ├── evaluation.py         # Metrics and evaluation reporting
│   └── api.py                # FastAPI serving endpoint
├── configs/
│   └── config.yaml           # Model and training configuration
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Train with synthetic data
python -m src.tft_model --config configs/config.yaml

# Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint     | Description                           |
|--------|-------------|---------------------------------------|
| POST   | /predict    | Multi-horizon RUL forecast            |
| POST   | /interpret  | Attention weights and variable importance |
| GET    | /health     | Service health check                  |

## Configuration

Model hyperparameters, data settings, and training configuration are centralized in `configs/config.yaml`. Key parameters include:

- `hidden_size`: Width of the GRN and attention layers
- `attention_head_size`: Number of attention heads
- `max_encoder_length`: Historical context window
- `max_prediction_length`: Number of future horizons to predict
- `loss`: Quantile loss for probabilistic forecasting

## Evaluation

No results are published in this repository. `src/evaluation.py` provides `TFTEvaluator`, which computes RMSE, MAE, MAPE and SMAPE per prediction horizon plus interval calibration, and `generate_report(actuals, predictions, output_dir="reports")` writes `metrics.json` and plots. It is a library class, not a CLI; call it from Python on the predictions produced by `TFTPredictor.predict`.

## License

MIT License
