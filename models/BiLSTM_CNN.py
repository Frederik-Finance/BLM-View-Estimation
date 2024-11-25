import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.volume import AccDistIndexIndicator
from ta.trend import EMAIndicator
import ta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Bidirectional, LSTM
from tensorflow.keras.losses import Huber
import joblib
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import logging
from ViewEstimation.models.BiLSTM_CNN_visualizations import (
    plot_training_history,
    plot_cumulative_returns_with_predictions,
    plot_rolling_and_cumulative_returns_with_intervals
)
import os

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    df['log_returns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    df.dropna(inplace=True)
    return df


def create_technical_indicators(df):
    df['SMA'] = ta.trend.sma_indicator(close=df['Adj Close'], window=14)
    df['EMA'] = ta.trend.ema_indicator(close=df['Adj Close'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(close=df['Adj Close'], window=50)
    macd = ta.trend.MACD(close=df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Adj Close'], window=14).rsi()
    adl = AccDistIndexIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        volume=df['Volume']
    ).acc_dist_index()
    ema_fast = EMAIndicator(close=adl, window=3).ema_indicator()
    ema_slow = EMAIndicator(close=adl, window=10).ema_indicator()
    df['ADOSC'] = ema_fast - ema_slow
    df['PSAR'] = ta.trend.PSARIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        step=0.02, 
        max_step=0.2
    ).psar()
    df['ADX'] = ta.trend.ADXIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14
    ).adx()
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14
    ).average_true_range()
    stoch_fast = ta.momentum.StochasticOscillator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14, 
        smooth_window=3
    )
    df['Stoch_Fast_K'] = stoch_fast.stoch()
    df['Stoch_Fast_D'] = stoch_fast.stoch_signal()
    stoch_slow = ta.momentum.StochasticOscillator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14, 
        smooth_window=3
    )
    df['Stoch_Slow_K'] = stoch_slow.stoch()
    df['Stoch_Slow_D'] = stoch_slow.stoch_signal()
    bollinger = ta.volatility.BollingerBands(
        close=df['Adj Close'], 
        window=20, 
        window_dev=2
    )
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()
    df['Williams_%R'] = ta.momentum.WilliamsRIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        lbp=14
    ).williams_r()
    df['CCI'] = ta.trend.CCIIndicator(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=20
    ).cci()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], 
        volume=df['Volume']
    ).on_balance_volume()
    ichimoku = ta.trend.IchimokuIndicator(
        high=df['High'], 
        low=df['Low'], 
        window1=9, 
        window2=26, 
        window3=52
    )
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    return df.dropna(axis=0)


from hurst import compute_Hc


def hurst_exponent(ts):
    if len(ts) < 20:
        return np.nan
    try:
        H, c, data = compute_Hc(ts, kind='price', simplified=True)
        return H
    except:
        return np.nan


def build_cnn_bilstm_model(input_shape, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x, training=True)
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x, training=True)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x, training=True)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(dropout_rate)(x, training=True)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=Huber())
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    non_zero_indices = y_test != 0
    if np.any(non_zero_indices):
        mape = np.mean(np.abs((y_test[non_zero_indices] - predictions[non_zero_indices]) / y_test[non_zero_indices])) * 100
    else:
        mape = np.nan
    return mae, rmse, mape


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def backtest_strategy(
    dates,
    actual_returns,
    predicted_returns,
    initial_capital=100000,
    transaction_cost=0.001,
    position_size=1.0,
    threshold=0.0
):
    capital = initial_capital
    portfolio_value = []
    positions = []
    returns = []
    signals = []
    for i in range(len(predicted_returns)):
        pred = predicted_returns[i]
        actual = actual_returns[i]
        date = dates[i]
        if pred > threshold:
            signal = 'Long'
            target_position = position_size
        elif pred < -threshold:
            signal = 'Short'
            target_position = -position_size
        else:
            signal = 'Hold'
            target_position = 0.0
        signals.append(signal)
        if len(positions) == 0:
            current_position = 0.0
        else:
            current_position = positions[-1]
        if signal == 'Hold':
            new_position = current_position
            trade_cost = 0.0
        else:
            if target_position != current_position:
                trade_cost = abs(target_position - current_position) * capital * transaction_cost
                capital -= trade_cost
                new_position = target_position
            else:
                trade_cost = 0.0
                new_position = current_position
        positions.append(new_position)
        portfolio_return = new_position * actual
        returns.append(portfolio_return)
        capital += portfolio_return * capital
        portfolio_value.append(capital)
    backtest_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Returns': predicted_returns,
        'Actual_Returns': actual_returns,
        'Signal': signals,
        'Position': positions,
        'Portfolio_Return': returns,
        'Portfolio_Value': portfolio_value
    })
    backtest_df.set_index('Date', inplace=True)
    total_return = (capital - initial_capital) / initial_capital * 100
    annualized_return = (1 + (capital / initial_capital - 1)) ** (252 / len(backtest_df)) - 1
    annualized_return *= 100
    annualized_volatility = backtest_df['Portfolio_Return'].std() * np.sqrt(252) * 100
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    max_drawdown = calculate_max_drawdown(backtest_df['Portfolio_Value'])
    results = {
        'Total Return (%)': total_return,
        'Annualized Return (%)': annualized_return,
        'Annualized Volatility (%)': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown (%)': max_drawdown,
        'Backtest DataFrame': backtest_df
    }
    plot_equity_curve(backtest_df)
    return results


def calculate_max_drawdown(portfolio_values):
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = np.max(drawdowns) * 100
    return max_drawdown


def plot_equity_curve(backtest_df):
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_df.index, backtest_df['Portfolio_Value'], label='Portfolio Value')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_sequences(features, target, sequence_length=7, forecast_horizon=1):
    X = []
    y = []
    for i in range(len(features) - sequence_length - forecast_horizon + 1):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length + forecast_horizon - 1])
    return np.array(X), np.array(y)


def main():
    df = load_and_prepare_data(r'/root/QaFamML/data/spy_data.csv')
    df = create_technical_indicators(df)
    selected_features = [
        'Adj Close', 'Volume', 'EMA', 'MACD_Diff', 'RSI',
        'ADX', 'ATR', 'OBV', 'CCI', 'Stoch_Slow_D'
    ]
    target_column = 'log_returns'
    df = df.dropna()
    X = df[selected_features].values
    y = df[target_column].values
    tscv = TimeSeriesSplit(n_splits=5)
    sequence_length = 30
    forecast_horizon = 6
    all_backtest_results = []
    for split_num, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Processing Split {split_num + 1}...")
        X_train_raw = X[train_index]
        y_train_raw = y[train_index]
        X_test_raw = X[test_index]
        y_test_raw = y[test_index]
        if len(X_train_raw) < sequence_length:
            print(f"Not enough data to create sequences for the training set in Split {split_num + 1}. Skipping this split.")
            continue
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_raw)
        X_test_scaled = scaler_X.transform(X_test_raw)
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length, forecast_horizon)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length, forecast_horizon)
        if len(X_test_seq) == 0:
            print(f"Not enough data to create sequences for the test set in Split {split_num + 1}. Skipping this split.")
            continue
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        cnn_bilstm_model = build_cnn_bilstm_model(input_shape)
        cnn_bilstm_history = cnn_bilstm_model.fit(
            X_train_seq, y_train_seq, 
            epochs=50, 
            batch_size=64, 
            validation_split=0.1,
            shuffle=False
        )
        cnn_bilstm_mae, cnn_bilstm_rmse, cnn_bilstm_mape = evaluate_model(cnn_bilstm_model, X_test_seq, y_test_seq)
        print(f"Split {split_num + 1} - CNN-BiLSTM Model Evaluation:")
        print(f"MAE: {cnn_bilstm_mae:.6f}")
        print(f"RMSE: {cnn_bilstm_rmse:.6f}")
        print(f"MAPE: {cnn_bilstm_mape:.2f}%\n")
        plot_training_history(cnn_bilstm_history)
        predictions = cnn_bilstm_model.predict(X_test_seq).flatten()
        n_iterations = 50
        quantile = 0.05
        predictions_list = []
        for _ in range(n_iterations):
            preds = cnn_bilstm_model.predict(X_test_seq, verbose=0).flatten()
            predictions_list.append(preds)
        predictions_array = np.array(predictions_list)
        predictions_mean = predictions_array.mean(axis=0)
        lower_bounds = np.quantile(predictions_array, quantile, axis=0)
        upper_bounds = np.quantile(predictions_array, 1 - quantile, axis=0)
        y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        predictions_mean_original = scaler_y.inverse_transform(predictions_mean.reshape(-1, 1)).flatten()
        lower_bounds_original = scaler_y.inverse_transform(lower_bounds.reshape(-1, 1)).flatten()
        upper_bounds_original = scaler_y.inverse_transform(upper_bounds.reshape(-1, 1)).flatten()
        cumulative_actual_returns = np.cumsum(y_test_original)
        test_dates_full = df.index[test_index]
        adjusted_start = sequence_length + forecast_horizon - 1
        test_dates = test_dates_full[adjusted_start : adjusted_start + len(y_test_original)]
        min_length = min(
            len(test_dates),
            len(cumulative_actual_returns),
            len(predictions_mean_original),
            len(lower_bounds_original),
            len(upper_bounds_original)
        )
        test_dates = test_dates[:min_length]
        cumulative_actual_returns = cumulative_actual_returns[:min_length]
        predictions_mean_original = predictions_mean_original[:min_length]
        lower_bounds_original = lower_bounds_original[:min_length]
        upper_bounds_original = upper_bounds_original[:min_length]
        backtest_dates = test_dates
        backtest_actual_returns = y_test_original[:min_length]
        backtest_predicted_returns = predictions_mean_original[:min_length]
        backtest_results = backtest_strategy(
            dates=backtest_dates,
            actual_returns=backtest_actual_returns,
            predicted_returns=backtest_predicted_returns,
            initial_capital=100000,
            transaction_cost=0.001,
            position_size=1.0,
            threshold=0.0
        )
        all_backtest_results.append({
            'Split': split_num + 1,
            'Metrics': backtest_results
        })
    for result in all_backtest_results:
        print(f"Split {result['Split']} Backtest Results:")
        for metric, value in result['Metrics'].items():
            if metric != 'Backtest DataFrame':
                print(f"  {metric}: {value}")
        print("\n")


    
if __name__ == "__main__":
    main()
