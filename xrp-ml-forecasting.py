import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Optional - install PyTorch and GluonTS if using DeepAR directly
# !pip install torch gluonts mxnet

def train_xgboost_model(data_path, target_column='tx_count', forecast_horizon=24, test_size=0.2):
    """
    Train an XGBoost model for time series forecasting
    
    Args:
        data_path: Path to the prepared CSV file
        target_column: Target column to predict
        forecast_horizon: How many steps ahead to forecast
        test_size: Proportion of data to use for testing
    
    Returns:
        Trained model, test predictions, and performance metrics
    """
    # Load the prepared data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Drop NaN values (from lag features)
    df = df.dropna()
    
    # Split data ensuring time order is preserved
    train_size = int(len(df) * (1 - test_size))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Prepare features and target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.01, 
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric='rmse',  # Metric in constructor
        early_stopping_rounds=50,  # Early stopping in constructor
        random_state=42
    )
    
    # Train with eval set
    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    model.fit(
        X_train_scaled, y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    print(f"XGBoost Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Save model and scaler
    model_dir = os.path.dirname(data_path)
    model_name = os.path.basename(data_path).replace('.csv', '')
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(model_dir, '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = f"{models_dir}/{model_name}_xgboost_model.pkl"
    print(f"Saving XGBoost model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler for later use in prediction
    scaler_path = f"{models_dir}/{model_name}_scaler.pkl"
    print(f"Saving scaler to {scaler_path}")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names for prediction
    feature_names = X_train.columns.tolist()
    feature_names_path = f"{models_dir}/{model_name}_feature_names.json"
    print(f"Saving feature names to {feature_names_path}")
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Save feature importance
    feature_importance.to_csv(f"{models_dir}/{model_name}_feature_importance.csv", index=False)
    
    # Visualize actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, y_test, label='Actual')
    plt.plot(test_data.index, y_pred, label='Predicted')
    plt.title('XGBoost: Actual vs Predicted Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Transaction Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}_xgboost_predictions.png")
    
    return {
        'model': model,
        'predictions': y_pred,
        'actual': y_test,
        'metrics': metrics,
        'feature_importance': feature_importance
    }

def prepare_sequences(df, target_col, seq_length, forecast_horizon):
    """
    Prepare sequences for LSTM/GRU models
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        seq_length: Length of input sequence
        forecast_horizon: How many steps ahead to forecast
    
    Returns:
        X and y for training deep learning models
    """
    X, y = [], []
    
    for i in range(len(df) - seq_length - forecast_horizon + 1):
        X.append(df.iloc[i:(i + seq_length)].values)
        y.append(df.iloc[i + seq_length:i + seq_length + forecast_horizon][target_col].values)
    
    return np.array(X), np.array(y)

def train_lstm_model(data_path, target_column='tx_count', seq_length=24, forecast_horizon=1, test_size=0.2):
    """
    Train an LSTM model for time series forecasting
    
    Args:
        data_path: Path to the prepared CSV file
        target_column: Target column to predict
        seq_length: Number of time steps to use as input
        forecast_horizon: How many steps ahead to forecast
        test_size: Proportion of data to use for testing
    
    Returns:
        Trained model, test predictions, and performance metrics
    """
    # Load the prepared data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Drop NaN values
    df = df.dropna()
    
    # Scale all features for deep learning
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Save target scaler for later inverse transformation
    target_scaler = StandardScaler()
    target_scaler.fit_transform(df[[target_column]])
    
    # Prepare sequences
    X, y = prepare_sequences(df_scaled, target_column, seq_length, forecast_horizon)
    
    # Split data ensuring time order is preserved
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=(seq_length, X.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Create callbacks
    model_dir = os.path.dirname(data_path)
    model_name = os.path.basename(data_path).replace('.csv', '')
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(model_dir, '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Update model file extension to .keras for newer Keras versions
    checkpoint = ModelCheckpoint(
        f"{models_dir}/{model_name}_lstm_model.keras",
        save_best_only=True,
        monitor='val_loss'
    )
    
    early_stopping = EarlyStopping(
        patience=20,
        restore_best_weights=True,
        monitor='val_loss'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform for plotting and metrics
    # We need to be careful about shapes here
    if forecast_horizon == 1:
        y_test_2d = y_test.reshape(-1, 1)
        y_pred_2d = y_pred.reshape(-1, 1)
    else:
        # For multi-step forecasting, we need to handle the shapes differently
        # This is a simplified approach - might need adjustment based on your forecast_horizon
        y_test_2d = y_test
        y_pred_2d = y_pred
    
    # Get the number of samples
    n_samples = y_test_2d.shape[0]
    
    # Create dummy dataframes with all zeros except target column
    dummy_test = np.zeros((n_samples, len(df.columns)))
    dummy_pred = np.zeros((n_samples, len(df.columns)))
    
    # Get the index of the target column
    target_idx = df.columns.get_loc(target_column)
    
    # Fill in the target values - adjust slice if multi-step
    if forecast_horizon == 1:
        dummy_test[:, target_idx] = y_test_2d.ravel()
        dummy_pred[:, target_idx] = y_pred_2d.ravel()
    else:
        # For multi-step, we might need to use only the first prediction
        # or aggregate the predictions depending on how you want to evaluate
        dummy_test[:, target_idx] = y_test_2d[:, 0]  # Use first step of multi-step forecast
        dummy_pred[:, target_idx] = y_pred_2d[:, 0]
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(dummy_test)[:, target_idx]
    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, target_idx]
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test_inv, y_pred_inv),
        'RMSE': np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
        'R2': r2_score(y_test_inv, y_pred_inv)
    }
    
    print(f"LSTM Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}_lstm_training_history.png")
    
    # Get appropriate test dates
    test_dates = df.index[train_size + seq_length:train_size + seq_length + len(y_test)]
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label='Actual')
    plt.plot(test_dates, y_pred_inv, label='Predicted')
    plt.title('LSTM: Actual vs Predicted Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Transaction Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_dir}/{model_name}_lstm_predictions.png")
    
    return {
        'model': model,
        'predictions': y_pred_inv,
        'actual': y_test_inv,
        'metrics': metrics,
        'history': history.history
    }

def implement_deepar(data_path, target_column='tx_count', forecast_horizon=24):
    """
    Implementing DeepAR (Amazon SageMaker) or PyTorch alternative
    
    Note: This is a basic implementation pattern - detailed implementation 
    depends on whether you're using AWS SageMaker or local PyTorch
    """
    print("DeepAR implementation requires either AWS SageMaker or local PyTorch+GluonTS")
    print("This function provides guidance and sample code for both approaches")
    
    # Path to the JSON file containing DeepAR formatted data
    json_path = data_path.replace('.csv', '.json')
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        print("Please ensure you have DeepAR formatted JSON file")
        return
    
    print("\n==== OPTION 1: Using AWS SageMaker ====")
    print("""
    # Sample code for AWS SageMaker implementation
    
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    role = get_execution_role()
    region = sagemaker.Session().boto_region_name
    
    # Specify DeepAR container
    container = get_image_uri(region, 'forecasting-deepar')
    
    # Set up hyperparameters
    hyperparameters = {
        "time_freq": "H",  # For hourly data, use "D" for daily
        "context_length": "72",  # Use ~3x your seasonal pattern
        "prediction_length": str(forecast_horizon),
        "epochs": "400",
        "early_stopping_patience": "40"
    }
    
    # Configure the estimator
    estimator = sagemaker.estimator.Estimator(
        container,
        role,
        train_instance_count=1,
        train_instance_type='ml.c5.xlarge',
        output_path='s3://<your-bucket>/output',
        sagemaker_session=sagemaker.Session(),
        hyperparameters=hyperparameters
    )
    
    # Prepare data channels
    data_channels = {
        "train": "s3://<your-bucket>/train/",
        "test": "s3://<your-bucket>/test/"
    }
    
    # Train the model
    estimator.fit(inputs=data_channels)
    
    # Create a predictor
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )
    
    # Make predictions
    # ...
    """)
    
    print("\n==== OPTION 2: Using PyTorch and GluonTS locally ====")
    print("""
    # Sample code for local GluonTS implementation
    
    from gluonts.dataset.common import ListDataset
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.mx.trainer import Trainer
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the data from JSON
    with open('data_path.json', 'r') as f:
        data = json.load(f)
    
    # Convert to GluonTS dataset format
    train_ds = ListDataset(
        [{"target": data["target"][:-forecast_horizon], "start": data["start"]}],
        freq="H"  # Use "D" for daily data
    )
    
    test_ds = ListDataset(
        [{"target": data["target"], "start": data["start"]}],
        freq="H"
    )
    
    # Create the estimator
    estimator = DeepAREstimator(
        freq="H",
        prediction_length=forecast_horizon,
        trainer=Trainer(
            epochs=100,
            learning_rate=1e-3,
            patience=10,
            num_batches_per_epoch=100
        )
    )
    
    # Train the model
    predictor = estimator.train(train_ds)
    
    # Make predictions
    forecast_it = predictor.predict(test_ds)
    forecasts = list(forecast_it)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    for forecast in forecasts:
        forecast.plot()
    plt.plot(data["target"])
    plt.legend(['Forecast', 'Actual'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("deepar_forecast.png")
    """)
    
    return

def main(data_dir="./output", 
        hourly_file="xgboost_hourly.csv", 
        daily_file="xgboost_daily.csv",
        forecast_horizon_hourly=24,  # 24 hours ahead
        forecast_horizon_daily=7):   # 7 days ahead
    """
    Main function to train and evaluate models
    
    Args:
        data_dir: Directory with prepared data
        hourly_file: Filename for hourly data
        daily_file: Filename for daily data
        forecast_horizon_hourly: How many hours to forecast
        forecast_horizon_daily: How many days to forecast
    """
    # Construct full paths
    hourly_path = os.path.join(data_dir, hourly_file)
    daily_path = os.path.join(data_dir, daily_file)
    
    # Check if files exist
    if not os.path.exists(hourly_path):
        print(f"File not found: {hourly_path}")
        return
    
    if not os.path.exists(daily_path):
        print(f"File not found: {daily_path}")
        return
    
    # Train XGBoost models
    print("\n===== Training XGBoost on Hourly Data =====")
    xgb_hourly_results = train_xgboost_model(
        hourly_path,
        forecast_horizon=forecast_horizon_hourly
    )
    
    print("\n===== Training XGBoost on Daily Data =====")
    xgb_daily_results = train_xgboost_model(
        daily_path,
        forecast_horizon=forecast_horizon_daily
    )
    
    # Train LSTM models
    print("\n===== Training LSTM on Hourly Data =====")
    lstm_hourly_results = train_lstm_model(
        hourly_path,
        seq_length=48,  # Use 2 days of history
        forecast_horizon=forecast_horizon_hourly
    )
    
    print("\n===== Training LSTM on Daily Data =====")
    lstm_daily_results = train_lstm_model(
        daily_path,
        seq_length=30,  # Use 1 month of history
        forecast_horizon=forecast_horizon_daily
    )
    
    # Example of DeepAR implementation
    print("\n===== DeepAR Implementation =====")
    implement_deepar(hourly_path, forecast_horizon=forecast_horizon_hourly)
    
    # Compare models
    print("\n===== Model Comparison =====")
    models = {
        'XGBoost (Hourly)': xgb_hourly_results['metrics'],
        'XGBoost (Daily)': xgb_daily_results['metrics'],
        'LSTM (Hourly)': lstm_hourly_results['metrics'],
        'LSTM (Daily)': lstm_daily_results['metrics']
    }
    
    comparison_df = pd.DataFrame(models).T
    print(comparison_df)
    
    # Save comparison results
    comparison_df.to_csv(f"{data_dir}/model_comparison.csv")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    comparison_df['RMSE'].plot(kind='bar')
    plt.title('Model Comparison - RMSE')
    plt.ylabel('RMSE (Root Mean Squared Error)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{data_dir}/model_comparison_rmse.png")
    
    print(f"\nAll models trained and evaluated. Results saved to {data_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models for XRP transaction forecasting')
    parser.add_argument('--data_dir', default='./output', help='Directory with prepared data')
    parser.add_argument('--hourly_file', default='xgboost_hourly.csv', help='Filename for hourly data')
    parser.add_argument('--daily_file', default='xgboost_daily.csv', help='Filename for daily data')
    parser.add_argument('--forecast_horizon_hourly', type=int, default=24, help='How many hours to forecast')
    parser.add_argument('--forecast_horizon_daily', type=int, default=7, help='How many days to forecast')
    
    args = parser.parse_args()
    
    main(
        args.data_dir,
        args.hourly_file,
        args.daily_file,
        args.forecast_horizon_hourly,
        args.forecast_horizon_daily
    )