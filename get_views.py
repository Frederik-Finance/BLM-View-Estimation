import pandas as pd
import numpy as np
import logging
from models.BiLSTM_CNN_Class import BiLSTM_CNN
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pmdarima as pm
import warnings
import os
from tensorflow.keras import backend as K
import numpy as np

import numpy as np
import gc


warnings.filterwarnings("ignore")


def get_all_etfs(directory_path):
    all_current_etfs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            truncated_fn = filename.split('.')[0]
            all_current_etfs.append(truncated_fn)
    return all_current_etfs


logging.getLogger('tensorflow').setLevel(logging.ERROR)


def load_and_prepare_data(etf_name, dir_path='./data/', date_cutoff=None):
    df = pd.read_csv(f'{dir_path}{etf_name}.csv', parse_dates=['Date'])
    df = df.set_index('Date')

    df = df.sort_index()

    if date_cutoff:
        df = df.loc[:date_cutoff]

    df['log_returns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    df['raw_returns'] = (df['Adj Close'] - df['Adj Close'].shift(1)) / df['Adj Close'].shift(1)
    print(df.index[-1])
    df.dropna(inplace=True)

    return df


def load_and_prepare_etf_data(etf_list, dir_path=r'/root/QaFamML/ViewEstimation/etfs/', date_cutoff=None):
    etf_data = {}
    for etf in etf_list:
        try:
            df = load_and_prepare_data(etf, dir_path=dir_path, date_cutoff=date_cutoff)
            df = BiLSTM_CNN.create_technical_indicators(df)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            etf_data[etf] = df
        except Exception as e:
            print(f"Error loading data for ETF {etf}: {e}")
    return etf_data


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff = np.where(denominator == 0, 0.0, diff)
    return np.mean(diff) * 100


def train_and_predict_arima(forecast_horizon, etf_data, etf_list, oos_run_length, target_column='Adj Close', max_lookback_period=1000):
    etf_forecasted_returns = {}
    etf_rmse = {}
    etf_smape = {}
    arima_predictions_dfs = {}

    arima_predictions_dir = '/root/QaFamML/ViewEstimation/performances/arima'

    for etf in etf_list:
        df = etf_data[etf].dropna()

        if len(df) < oos_run_length + forecast_horizon:
            print(f"Not enough data to train ARIMA model for ETF {etf}. Skipping.")
            continue

        total_slice = max_lookback_period + oos_run_length
        start_idx = df.index[-total_slice] if len(df) > total_slice else df.index[0]
        train_df = df.loc[start_idx:df.index[-oos_run_length-1]]
        test_df = df.iloc[-oos_run_length:]

        y_train = train_df[target_column]
        y_test = test_df[target_column]

        try:
            model = pm.auto_arima(
                y_train,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3,
                max_q=3,
                max_d=2
            )
        except Exception as e:
            print(f"Error fitting ARIMA model for ETF {etf}: {e}")
            continue

        predictions = []
        actuals = []
        prediction_dates = []

        for t in range(0, len(y_test) - forecast_horizon + 1, forecast_horizon):
            forecast = model.predict(n_periods=forecast_horizon)
            predictions.extend(forecast)
            actual_values = y_test.iloc[t:t + forecast_horizon]
            actuals.extend(actual_values)
            prediction_dates.extend(actual_values.index)
            model.update(actual_values)

        min_length = min(len(actuals), len(predictions))
        actuals = actuals[:min_length]
        predictions = predictions[:min_length]
        prediction_dates = prediction_dates[:min_length]

        predictions_df = pd.DataFrame({
            'actuals': actuals,
            'predictions': predictions
        }, index=prediction_dates)

        predictions_df.to_csv(os.path.join(arima_predictions_dir, f'{etf}.csv'))
        arima_predictions_dfs[etf] = predictions_df

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        smape = symmetric_mean_absolute_percentage_error(np.array(actuals), np.array(predictions))
        etf_rmse[etf] = rmse
        etf_smape[etf] = smape

        full_data = df[target_column]

        try:
            model_full = pm.auto_arima(
                full_data,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3,
                max_q=3,
                max_d=2
            )
        except Exception as e:
            print(f"Error fitting ARIMA model on full data for ETF {etf}: {e}")
            continue

        future_forecast = model_full.predict(n_periods=forecast_horizon)

        last_value = full_data.iloc[-1]
        forecasted_return = (future_forecast.iloc[-1] / last_value) - 1

        etf_forecasted_returns[etf] = forecasted_return

    return etf_forecasted_returns, etf_rmse, etf_smape


def train_and_predict_lstm(forecast_horizon, etf_data, etf_list, sequence_length, oos_run_length, selected_features, target_column, max_lookback_period=5000, save_performance=True):
    etf_predictions = {}
    prediction_dates = {}
    etf_rmse = {}
    etf_smape = {}

    for etf in etf_list:
        df = etf_data[etf].dropna()

        if len(df) < sequence_length + oos_run_length:
            print(f"Not enough data to train model for ETF {etf}. Skipping.")
            continue

        total_slice = max_lookback_period + oos_run_length
        start_idx = df.index[-total_slice] if len(df) > total_slice else df.index[0]
        train_df = df.loc[start_idx:df.index[-oos_run_length-1]]
        test_df = df.iloc[-oos_run_length:]

        model = BiLSTM_CNN(
            feature_columns=selected_features,
            target_column=target_column,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            dropout_rate=0.1,
            batch_size=64,
            epochs=100,
            validation_split=0.1,
            use_early_stopping=False,
            random_seed=42
        )

        history = model.train(train_df)
        if history is None:
            print(f"Skipping ETF {etf} due to insufficient training data.")
            continue

        predictions, dates = model.predict(test_df)
        print(f"Number of predictions: {len(predictions)}")
        print(f"OOS run length: {oos_run_length}")
        print(f"Length of test_df: {len(test_df)}")

        mae, rmse, smape = model.evaluate(test_df)

        predictions_df = pd.DataFrame(data={'predictions': predictions}, index=dates)

        test_df = test_df.merge(predictions_df, left_index=True, right_index=True, how='left')

        test_df = test_df.iloc[sequence_length:]

        test_df.to_csv(rf'/root/QaFamML/ViewEstimation/performances/lstm/{etf}.csv')

        if 'predictions' in test_df.columns:
            test_df.dropna(subset=[target_column, 'predictions'], inplace=True)

            rmse = np.sqrt(mean_squared_error(test_df[target_column], test_df['predictions']))
            smape = symmetric_mean_absolute_percentage_error(test_df[target_column], test_df['predictions'])
            etf_rmse[etf] = rmse
            etf_smape[etf] = smape

            prediction_dates[etf] = test_df.index[-1]
            last_value = test_df[target_column].iloc[-1]
            forecasted_return = (test_df['predictions'].iloc[-1] / last_value) - 1
            etf_predictions[etf] = forecasted_return
            print(train_df.columns)
        else:
            print(f"No prediction made for ETF {etf}.")

        K.clear_session()
        del model, history, predictions, dates, predictions_df

    return etf_predictions, prediction_dates, etf_rmse, etf_smape


def compute_p_q_matrices(etf_predictions, view_type='absolute'):

    assets = list(etf_predictions.keys())
    N = len(assets)

    if view_type == 'absolute':
        P = np.eye(N)
        Q = np.array([etf_predictions[asset] for asset in assets])
    elif view_type == 'relative':
        views = []
        Q_list = []
        for i in range(N):
            for j in range(i+1, N):
                delta_return = etf_predictions[assets[i]] - etf_predictions[assets[j]]
                if delta_return != 0:
                    P_row = np.zeros(N)
                    P_row[i] = 1
                    P_row[j] = -1
                    views.append(P_row)
                    Q_list.append(delta_return)
        P = np.array(views)
        Q = np.array(Q_list)
    else:
        raise ValueError("view_type must be 'absolute' or 'relative'")
    return P, Q


def get_model_forecasts(oos_run_length, forecast_horizon=1, date_cutoff=None, etf_list=['TLT', 'XLK'], model_to_choose='both',
                        lstm_hyperparameters={'sequence_length': 7, 'target_column': 'Adj Close',
                                              'selected_features': ['Volume', 'EMA', 'MACD_Diff', 'RSI', 'ADX', 'ATR', 'OBV', 'CCI', 'Stoch_Slow_D']})-> pd.DataFrame:
    etf_data = load_and_prepare_etf_data(etf_list, date_cutoff=date_cutoff)


    if model_to_choose in ['lstm', 'both']:
        selected_features = lstm_hyperparameters['selected_features']
        target_column = lstm_hyperparameters['target_column']
        sequence_length = lstm_hyperparameters['sequence_length']

        lstm_predictions, lstm_dates, lstm_rmse, lstm_smape = train_and_predict_lstm(
            forecast_horizon,
            etf_data, etf_list, sequence_length, oos_run_length, selected_features, target_column
        )

    if model_to_choose in ['arima', 'both']:
        arima_predictions, arima_rmse, arima_smape = train_and_predict_arima(
            forecast_horizon,
            etf_data, etf_list, oos_run_length, target_column='Adj Close', max_lookback_period=1000
        )

    data = []
    for etf in etf_list:
        row = {'ETF': etf}
        if model_to_choose in ['arima', 'both']:
            row['arima_pred'] = arima_predictions[etf]
            row['arima_rmse'] = arima_rmse[etf]
            row['arima_smape'] = arima_smape[etf]
        if model_to_choose in ['lstm', 'both']:
            row['lstm_pred'] = lstm_predictions[etf]
            row['lstm_rmse'] = lstm_rmse[etf]
            row['lstm_smape'] = lstm_smape[etf]
        data.append(row)

    forecasts_df = pd.DataFrame(data).set_index('ETF')
    return forecasts_df


def get_p_q(forecasts_df, criterion='rmse', view_type='absolute'):
    etf_predictions = {}
    for idx in forecasts_df.index:
        etf = forecasts_df.loc[idx, 'ETF']

        if criterion == 'rmse':
            arima_metric = forecasts_df.loc[idx, 'arima_rmse']
            lstm_metric = forecasts_df.loc[idx, 'lstm_rmse']
        elif criterion == 'smape':
            arima_metric = forecasts_df.loc[idx, 'arima_smape']
            lstm_metric = forecasts_df.loc[idx, 'lstm_smape']
        else:
            raise ValueError("Criterion must be 'rmse' or 'smape'")

        if arima_metric <= lstm_metric:
            selected_pred = forecasts_df.loc[idx, 'arima_pred']
        else:
            selected_pred = forecasts_df.loc[idx, 'lstm_pred']

        etf_predictions[etf] = selected_pred

    print(etf_predictions)
    P, Q = compute_p_q_matrices(etf_predictions, view_type=view_type)

    return P, Q


def run_forecasts_over_cutoffs(cutoff_dates, oos_run_length, forecast_horizon, etf_list, model_to_choose, lstm_hyperparameters, criterion='rmse'):
    results = {}
    for date_cutoff in cutoff_dates:
        forecasts_df = get_model_forecasts(
            oos_run_length=oos_run_length,
            forecast_horizon=forecast_horizon,
            date_cutoff=date_cutoff,
            etf_list=etf_list,
            model_to_choose=model_to_choose,
            lstm_hyperparameters=lstm_hyperparameters        )

        forecasts_df.to_csv(f'/root/QaFamML/ViewEstimation/forecasts/{date_cutoff}_forecast.csv')
        del forecasts_df
        gc.collect()

    return None  


def main():
    oos_run_length = 100
    forecast_horizon = 1

    directory_path = r'/root/QaFamML/ViewEstimation/etfs'
    etf_list = get_all_etfs(directory_path)
    print(len(etf_list))
    model_to_choose = 'both'
    lstm_hyperparameters = {
        'sequence_length': 7,
        'target_column': 'Adj Close',
        'selected_features': 
        ['Open', 'High', 'Low', 'Close', 'SMA', 'EMA', 'MACD', 'RSI', 'ADOSC', 'PSAR', 
                 'ADX', 'ATR', 'Stoch_Fast_K', 'Stoch_Fast_D', 'Stoch_Slow_K', 'Stoch_Slow_D', 
                 'Bollinger_High', 'Bollinger_Low', 'Williams_%R', 'CCI', 'OBV', 'Ichimoku_A', 'Ichimoku_B']
    }

    regime_df = pd.read_csv(r'/root/QaFamML/ViewEstimation/regimes.csv')

    cutoff_dates = [
    "2017-04-30 00:00:00",
    "2017-07-31 00:00:00",
    "2017-10-31 00:00:00",
    "2018-01-31 00:00:00",
    "2018-04-30 00:00:00",
    "2018-07-31 00:00:00",
    "2018-10-31 00:00:00",
    "2019-01-31 00:00:00",
    "2019-04-30 00:00:00",
    "2019-07-31 00:00:00",
    "2019-10-31 00:00:00",
    "2020-01-31 00:00:00",
    "2020-04-30 00:00:00",
    "2020-07-31 00:00:00",
    "2020-10-31 00:00:00",
    "2021-01-31 00:00:00",
    "2021-04-30 00:00:00",
    "2021-07-31 00:00:00",
    "2021-10-31 00:00:00",
    "2022-01-31 00:00:00",
    "2022-04-30 00:00:00",
    "2022-07-31 00:00:00",
    "2022-10-31 00:00:00",
    "2023-01-31 00:00:00",
    "2023-04-30 00:00:00",
    "2023-07-31 00:00:00",
    "2023-10-31 00:00:00"
    ]

    results = run_forecasts_over_cutoffs(
        cutoff_dates=cutoff_dates,
        oos_run_length=oos_run_length,
        forecast_horizon=forecast_horizon,
        etf_list=etf_list,
        model_to_choose=model_to_choose,
        lstm_hyperparameters=lstm_hyperparameters    )

    print(results)


if __name__ == "__main__":
    main()
