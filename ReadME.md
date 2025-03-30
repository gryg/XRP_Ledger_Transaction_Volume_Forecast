# XRP Transaction Volume Forecasting

This repository contains an end-to-end pipeline for analyzing and forecasting XRP transaction volume using multiple machine learning models.

## Project Overview

The XRP Transaction Volume Forecasting pipeline processes raw XRP ledger data, creates time series features, trains multiple forecasting models, evaluates their performance, and generates forecasts. The project compares the effectiveness of XGBoost and LSTM models at both hourly and daily prediction intervals.

## Current Progress

Based on the evaluation metrics, XGBoost has significantly outperformed LSTM for hourly forecasting:

```
===== Model Comparison =====
                         MAE         RMSE        R2
XGBoost (Hourly)    3.785696     5.094750  0.703324
XGBoost (Daily)   690.160006  2931.637085 -0.026138
LSTM (Hourly)       5.693151     7.447152  0.364917
LSTM (Daily)      146.426293   175.645594 -0.052365
```

Key findings:
- XGBoost performs well for hourly predictions with an R² of 0.70
- Both models struggle with daily predictions
- The hourly XGBoost forecast has significantly lower error rates (MAE & RMSE)
- In deployment, XGBoost achieves much better accuracy than LSTM or ensemble methods:
  - XGBoost Hourly MAE: 66.23, MAPE: 7.03%
  - LSTM Hourly MAE: 1001.18, MAPE: 99.82%

### Model Visualizations

#### LSTM Models
![LSTM Daily Predictions](./xrp_results/processed_data/lstm_daily_predictions.png)
*Figure 1: LSTM model predictions vs actual transaction volume (daily)*

![LSTM Daily Training](./xrp_results/processed_data/lstm_daily_training_history.png)
*Figure 2: LSTM model training and validation loss (daily)*

![LSTM Hourly Predictions](./xrp_results/processed_data/lstm_hourly_predictions.png)
*Figure 3: LSTM model predictions vs actual transaction volume (hourly)*

![LSTM Hourly Training](./xrp_results/processed_data/lstm_hourly_training_history.png)
*Figure 4: LSTM model training and validation loss (hourly)*

#### XGBoost Models
![XGBoost Daily Predictions](./xrp_results/processed_data/xgboost_daily_predictions.png)
*Figure 5: XGBoost model predictions vs actual transaction volume (daily)*

![XGBoost Hourly Predictions](./xrp_results/processed_data/xgboost_hourly_predictions.png)
*Figure 6: XGBoost model predictions vs actual transaction volume (hourly)*

## Project Structure

```
.
├── xrp-data-processor.py     # Data processing and feature engineering
├── xrp-ml-forecasting.py     # Model training and evaluation
├── xrp-forecast-deployment.py # Model deployment and prediction
├── xrp-pipeline.py           # End-to-end pipeline orchestration
├── xrp_results/              # Output directory
    ├── processed_data/       # Processed datasets
    ├── models/               # Trained models
    ├── forecasts/            # Generated forecasts
    └── pipeline_report.html  # HTML summary report
```

## Next Steps

1. **Model Tuning**: Improve XGBoost hyperparameters for both hourly and daily predictions
2. **Feature Engineering**: Test additional time-based features to improve daily predictions
3. **Ensemble Methods**: Develop better ensemble methods with appropriate model weighting
4. **Visualization Tools**: Create additional visualization dashboards for monitoring
5. **DeepAR**: Implement DeepAR for time series forecasting
<!-- 5. **API Development**: Create an API for real-time forecasting -->

## Getting Started

To run the pipeline:

```bash
python xrp-pipeline.py --data_path /path/to/xrp_data.csv --output_dir ./xrp_results
```

For individual steps:

```bash
# Data processing
python xrp-data-processor.py --filepath /path/to/xrp_data.csv --output ./xrp_results/processed_data

# Model training
python xrp-ml-forecasting.py --data_dir ./xrp_results/processed_data

# Model evaluation
python xrp-forecast-deployment.py --model_dir ./xrp_results/models --data_file ./xrp_results/processed_data/hourly_transactions.csv --output_dir ./xrp_results/forecasts/hourly
```