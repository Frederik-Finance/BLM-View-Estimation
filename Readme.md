
# Performance Analysis & Future Improvements

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/Frederik-Finance/BLM-View-Estimation)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Model Performance & Results](#model-performance--results)
  - [Training Methodology](#training-methodology)
  - [Model Comparison: LSTM vs ARIMA](#model-comparison-lstm-vs-arima)
  - [LSTM Model Deep Dive](#lstm-model-deep-dive)
    - [Consumer Staples ETF (KXI) Performance](#consumer-staples-etf-kxi-performance)
    - [Technology ETF (VGT) Performance](#technology-etf-vgt-performance)
  - [Strategy Performance](#strategy-performance)
  - [Black-Litterman Integration Results](#black-litterman-integration-results)
  - [LSTM Performance Reflection](#lstm-performance-reflection)
- [Critical Analysis & Reflections](#critical-analysis--reflections)
  - [Strengths](#strengths)
- [Key Findings](#key-findings)
- [Areas for Improvement](#areas-for-improvement)
- [Modularization Recommendations](#modularization-recommendations)
- [License](#license)

## Project Overview
The **Performance Analysis & Future Improvements** project provides an in-depth evaluation of various financial models, focusing on their performance, strengths, and areas for enhancement. This analysis includes the implementation and comparison of LSTM and ARIMA models, integration with the Black-Litterman framework, and a reflection on the models' effectiveness in different market conditions.

## Features
- **Hybrid Model Architecture**
  - Bidirectional LSTM with CNN layers for complex pattern recognition.
  - ARIMA modeling for time series analysis.
  - Model selection based on performance metrics (RMSE/SMAPE).

- **Technical Analysis Integration**
  - 20+ technical indicators including EMA, MACD, RSI, Bollinger Bands.
  - Volume analysis with OBV and Accumulation/Distribution.
  - Momentum indicators with Stochastic Oscillator and Williams %R.

## Model Performance & Results

### Training Methodology
![Rolling Window Training](/images/rolling_window_training_process.png)

Our model employs a rolling window approach with three distinct periods:
- **Lookback Period**: Historical data used for model training.
- **Out-of-Sample Testing**: Previous quarter used for model validation.
- **Prediction**: Forward-looking forecast period.

### Model Comparison: LSTM vs ARIMA
The performance comparison between LSTM and ARIMA models across various ETFs reveals interesting patterns:

![Performance Metrics](/images/spy_table.png)

**Key Findings:**
- **ARIMA** generally shows lower RMSE values, suggesting better accuracy in point predictions.
- **Performance Differences Across ETF Categories:**
  - **Fixed Income ETFs** (e.g., TLT, IEF): Both models show stable performance.
  - **Technology ETFs** (e.g., VGT, XLK): Higher volatility reflected in larger RMSE values.
  - **Sector ETFs**: Mixed results depending on sector characteristics.

### LSTM Model Deep Dive

#### Consumer Staples ETF (KXI) Performance
![KXI Performance](/images/lstm_kxi.png)
- **Directional Accuracy**: 58.7%
- **RMSE**: 2.31
- **MAPE**: 3.20%
- **Insights**: Strong performance in capturing trend direction with reasonable confidence intervals.

#### Technology ETF (VGT) Performance
![VGT Performance](/images/lstm_vgt.png)
- **Directional Accuracy**: 46.7%
- **RMSE**: 46.23
- **MAPE**: 8.62%
- **Insights**: Larger confidence intervals reflecting higher sector volatility.

### LSTM Performance Reflection

The relatively high volatility in LSTM predictions and higher errors stem from a key limitation in the approach, namely that this is modeled as a regression problem because continuous values are needed to construct Black-Litterman Model (BLM) views.

The inherent challenge with this approach is that it leads to the LSTM paying the most attention to market events that should receive the least attention: outlier events caused by extraneous factors. Since the LSTM only has price-based features, it lacks contextual information to learn from these external events, negatively impacting its performance.

While the implementation of a Huber loss function provides some mitigation, the core problem persists. Potential improvements could include:

1. Expanding the feature set with domain-specific features for each ETF
2. Modeling the LSTM as a classification problem in scenarios where continuous values aren't required
3. Incorporating external market event data to provide context for outlier movements

However, these enhancements were outside the scope of this project's initial implementation.


### Strategy Performance
![Strategy Performance](/images/blm_outperformance.png)

The ETF selection strategy shows significant outperformance versus the benchmark (SPY):
- **Cumulative Returns (Jul '17 - Apr '23):**
  - **Strategy**: ~160% return
  - **Benchmark**: ~80% return
- **Notable Outperformance During Market Volatility (2021-2023)**
- **Strong Yearly Returns:**
  - Particularly in 2022 with ~60% return vs benchmark -15%.

### Black-Litterman Integration Results
*Metrics and PNG pending inclusion.*
- **Impact on Portfolio Allocation**
- **Performance Improvements Over Baseline**


## Critical Analysis & Reflections

### Strengths
1. **Hybrid Approach**
   - Combines traditional time series methods (ARIMA) with deep learning (LSTM-CNN).
   - Leverages the strengths of both approaches.
   - Enables model selection based on performance metrics.

2. **Comprehensive Technical Analysis**
   - Employs rich feature engineering with 20+ technical indicators.
   - Considers volume and momentum.
   - Utilizes multiple timeframes for analysis.

3. **Robust Evaluation Framework**
   - Implements out-of-sample testing.
   - Utilizes multiple performance metrics.
   - Accounts for transaction costs.

## Key Findings

1. **Model Selection**
   - **ARIMA** shows superior performance for stable, low-volatility ETFs.
   - **LSTM** better captures complex patterns in high-volatility sectors.
   - **Hybrid Approach** enables optimal model selection based on ETF characteristics.

2. **Market Conditions**
   - **Robust Performance** across different market regimes.
   - **Strong During High Volatility Periods**.
   - **Consistent Outperformance** over the benchmark.

3. **Practical Applications**
   - **Real-World Applicability** demonstrated.
   - **Transaction Costs and Market Impact** considered.
   - **Robust Out-of-Sample Performance** validation.

## Areas for Improvement

1. **Model Architecture**
   - Add attention mechanisms to LSTM.
   - Implement transformer-based architectures.
   - Explore ensemble methods beyond the current hybrid approach.

2. **Feature Engineering**
   - Include market sentiment analysis.
   - Add macroeconomic indicators.
   - Consider alternative data sources.

3. **Risk Management**
   - Implement more sophisticated position sizing.
   - Add stop-loss mechanisms.
   - Consider volatility forecasting.

## Modularization Recommendations
```
View_Estimation/
├── etfs/                      # ETF data directory
├── forecasts/                 # Forecast results
├── images/
│   ├── blm_outperformance.png
│   ├── lstm_kxi.png
│   ├── lstm_vgt.png
│   ├── rolling_window_training_process.png
│   └── spy_table.png
├── models/
│   └── BiLSTM_CNN.py
├── performances/
├── create_performance_chart.py
├── get_views.py
└── README.md
```

**Reference:**
Barua, R., & Sharma, A. K. (2022). Dynamic Black Litterman portfolios with views derived via CNN-BiLSTM predictions. *Finance Research Letters*, 49, 103111. 

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any inquiries or feedback, please reach out to:

📧 **Email**: [fwoite.2024@mqf.smu.edu.sg](mailto:fwoite.2024@mqf.smu.edu.sg)
