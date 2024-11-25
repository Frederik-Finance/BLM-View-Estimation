import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy import stats

def calculate_directional_accuracy(actual, predicted):
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    correct_directions = np.sum(actual_direction == predicted_direction)
    total_predictions = len(actual_direction)
    directional_accuracy = (correct_directions / total_predictions) * 100
    return directional_accuracy, actual_direction, predicted_direction

def plot_performance(etf_dict, base_path, color_scheme='blue'):
    color_schemes = {
        'blue': {'main': 'blue', 'secondary': 'lightblue', 'error': 'navy'},
        'red': {'main': 'red', 'secondary': 'lightcoral', 'error': 'darkred'},
        'green': {'main': 'green', 'secondary': 'lightgreen', 'error': 'darkgreen'}
    }
    colors = color_schemes.get(color_scheme.lower(), color_schemes['blue'])
    plt.style.use('seaborn')
    
    for model, etfs in etf_dict.items():
        model_path = os.path.join(base_path, model)
        
        for etf in etfs:
            file_path = os.path.join(model_path, f"{etf}.csv")
            
            if not os.path.exists(file_path):
                print(f"File for {etf} in {model} model does not exist. Skipping.")
                continue
            
            data = pd.read_csv(file_path)
            if model == 'arima':
                data = data.rename(columns={'Unnamed: 0': 'Date', 'actuals': 'Adj Close'})
            
            data['Date'] = pd.to_datetime(data['Date'])
            dir_accuracy, actual_dir, pred_dir = calculate_directional_accuracy(
                data['Adj Close'].values, 
                data['predictions'].values
            )
            mse = mean_squared_error(data['Adj Close'], data['predictions'])
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((data['Adj Close'] - data['predictions']) / data['Adj Close'])) * 100
            residuals = data['Adj Close'] - data['predictions']
            std_error = np.std(residuals)
            z_score = stats.norm.ppf(0.975)
            margin_of_error = z_score * std_error
            lower_bound = data['predictions'] - margin_of_error
            upper_bound = data['predictions'] + margin_of_error
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5], hspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data['Date'], data['Adj Close'], label='Actual', 
                     color=colors['main'], linewidth=2, alpha=0.7)
            ax1.plot(data['Date'], data['predictions'], label='Forecast', 
                     color=colors['secondary'], linewidth=2, alpha=0.7, linestyle='--')
            ax1.fill_between(data['Date'], lower_bound, upper_bound,
                            alpha=0.2, color=colors['secondary'],
                            label='95% Confidence Interval')
            ax1.set_title(f'{etf} Performance Comparison ({model.upper()})\n'
                         f'RMSE: {rmse:.2f} | Dir. Accuracy: {dir_accuracy:.1f}% | MAPE: {mape:.2f}%',
                         fontsize=14, pad=20)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', frameon=True)
            ax1.set_ylabel('Price', fontsize=12)
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(data['Date'], residuals, color=colors['error'], 
                     alpha=0.7, label='Forecast Error')
            ax2.fill_between(data['Date'], residuals, 0, 
                            alpha=0.2, color=colors['error'])
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel('Error', fontsize=12)
            ax2.legend(loc='upper left')
            ax3 = fig.add_subplot(gs[2])
            sns.histplot(residuals, kde=True, ax=ax3, bins=30, 
                         color=colors['error'])
            ax3.set_title('Error Distribution', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            print(f"\nSummary Statistics for {etf} ({model.upper()}):")
            print(f"Directional Accuracy: {dir_accuracy:.1f}%")
            print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Prediction Standard Error: {std_error:.2f}")
            print(f"95% Confidence Interval: ±{margin_of_error:.2f}")

good_performance_dict = {
    'lstm': ['XLE', 'RXI', 'JXI', 'TLT', 'VNQ', 'IEF', 'VCR', 'VPU', 'XLF', 'IXG', 
             'PICK', 'VFH', 'XLV', 'XLRE', 'VGT', 'VIS', 'XLK', 'IXN', 'VDE', 'VDC', 
             'IYR', 'VOX', 'FILL', 'EXI', 'XLB', 'KXI', 'XLU', 'XLP', 'BIL', 'XLI', 
             'VAW', 'VHT', 'IXP', 'XLY', 'IXJ']
}

base_path = '/root/QaFamML/ViewEstimation/performances'
plot_performance(good_performance_dict, base_path, 'green')
