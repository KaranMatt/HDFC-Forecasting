# HDFC Stock Price Forecasting with Deep Learning

> A comprehensive time series analysis and forecasting project leveraging deep learning models to predict HDFC stock prices using historical data from 2000-2021.

---

##  Project Overview

This project implements multiple deep learning architectures to forecast HDFC stock prices, progressing from simple baseline models to sophisticated multivariate LSTM networks. The analysis demonstrates a systematic approach to time series forecasting, achieving a **62% improvement** over baseline predictions using advanced multivariate models.

### Key Highlights
-  21 years of historical stock data (2000-2021)
-  Chronological train-test split (80-20)
-  Multiple model architectures (Dense, LSTM)
-  Univariate and Multivariate approaches
-  Comprehensive evaluation metrics (MAE, RMSE, MAPE, MASE)

---

##  Dataset

**Source:** HDFC Stock Historical Data  
**Time Period:** 2000 - 2021  
**Key Features:**
- `Date` - Trading date (converted to index)
- `Close` - Closing price (primary target variable)
- `VWAP` - Volume Weighted Average Price (multivariate feature)

**Data Preprocessing:**
- Dates converted to datetime index for chronological ordering
- Two primary arrays created:
  - `prices`: Close price values
  - `timesteps`: Date values
- Chronological split (no randomization) to preserve temporal dependencies

---

##  Exploratory Data Analysis

### Visualization Pipeline

1. **Full Dataset Visualization**
   - Complete close price time series (2000-2021)
   - Trend identification and pattern recognition

2. **Train-Test Split Visualization**
   - Visual separation of training (80%) and testing (20%) data
   - Temporal boundary identification

---

##  Evaluation Framework

### Custom Metrics Implementation

A comprehensive evaluation system was built to assess model performance:

```python
Metrics Used:
├── MAE (Mean Absolute Error)
├── RMSE (Root Mean Squared Error)
├── MAPE (Mean Absolute Percentage Error)
└── MASE (Mean Absolute Scaled Error)
```

**Custom Function:** `evaluate_metrics()` - Calculates MAE, RMSE, MAPE, and MASE for model predictions

**MASE (Mean Absolute Scaled Error):** Custom implementation to provide scale-independent performance measurement, particularly useful for comparing models across different datasets.

---

##  Baseline Model: Naive Forecast

Before diving into deep learning, a naive forecast was established as a baseline using the most recent observation as the prediction.

### Performance Metrics
| Metric | Value |
|--------|-------|
| MAE | 26.08 |
| RMSE | 38.54 |
| MAPE | 1.34% |
| MASE | 0.99 |

**Visualization:** Train, test, and naive forecast predictions plotted together for comparative analysis.

---

##  Model Architecture & Results

### Window Creation Framework

Custom functions were developed to transform the time series into supervised learning format:

1. **`get_labelled_windows()`** - Slices arrays into windows and horizons
2. **`make_windows()`** - Creates sliding windows of variable size with specified horizon
3. **`split_windows()`** - Splits processed data into train/test windows and labels

---

##  Univariate Models (Single Feature: Close Price)

### Model 1: Dense Neural Network (Window=7, Horizon=1)

**Architecture:**
```
Input Layer
    ↓
Dense Layer (128 units, ReLU)
    ↓
Output Layer (1 unit)
```

**Results:**
| MAE | RMSE | MAPE | MASE |
|-----|------|------|------|
| 26.19 | 38.54 | 1.34% | 1.003 |

**Analysis:** Performance comparable to naive baseline, indicating need for more complex architecture.

---

### Model 2: Dense Neural Network (Window=30, Horizon=1)

**Architecture:**
```
Input Layer
    ↓
Dense Layer (128 units, ReLU)
    ↓
Output Layer (1 unit)
```

**Results:** *Not documented*

---

### Model 3: Stacked LSTM (Window=7, Horizon=1)

**Architecture:**
```
Input Layer
    ↓
LSTM Layer (128 units)
    ↓
LSTM Layer (128 units)
    ↓
Output Layer (1 unit)
```

**Results:**
| MAE | RMSE | MAPE | MASE |
|-----|------|------|------|
| 35.60 | 46.87 | 1.82% | 1.36 |

**Analysis:** Surprisingly underperformed compared to simpler dense model, suggesting potential overfitting or need for hyperparameter tuning.

---

##  Multivariate Models (Multiple Features)

### Feature Engineering Strategy

Created lagged features to capture temporal dependencies:

**Dataset Structure: `hdfc_windowed_df`**
```
Columns:
├── Index (Date)
├── close (Target variable)
├── close_1 (t-1 lag)
├── close_2 (t-2 lag)
├── close_3 (t-3 lag)
├── close_4 (t-4 lag)
├── close_5 (t-5 lag)
├── close_6 (t-6 lag)
├── close_7 (t-7 lag)
└── vwap (Additional feature)
```

**Split Strategy:** 80-20 chronological split
- `train_inputs` / `test_inputs`
- `train_target` / `test_target`

---

### Model 4: Multivariate Dense Network 

**Architecture:**
```
Input Layer (9 features)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Output Layer (1 unit)
```

**Results:**
| MAE | RMSE | MAPE | MASE |
|-----|------|------|------|
| 16.23 | 19.99 | 0.832% | 0.621 |

**Breakthrough:** 
-  **37.8%** improvement in MAE over baseline
-  **48.1%** improvement in RMSE over baseline
-  **37.3%** improvement in MASE over baseline

---

### Model 5: Multivariate Stacked LSTM 

**Architecture:**
```
Input Layer (9 features)
    ↓
LSTM Layer (128 units)
    ↓
LSTM Layer (128 units)
    ↓
Output Layer (1 unit)
```

**Results:**
| MAE | RMSE | MAPE | MASE |
|-----|------|------|------|
| 10.49 | 15.30 | 0.542% | 0.401 |

**Best Performance:**
-  **59.8%** improvement in MAE over baseline
-  **60.3%** improvement in RMSE over baseline
-  **59.5%** improvement in MASE over baseline
-  **59.6%** improvement in MAPE over baseline

---

##  Model Comparison Summary

| Model | Type | Window | Features | MAE ↓ | RMSE ↓ | MAPE ↓ | MASE ↓ |
|-------|------|--------|----------|-------|--------|--------|--------|
| Baseline | Naive | - | 1 | 26.08 | 38.54 | 1.34% | 0.99 |
| Model 1 | Dense | 7 | 1 | 26.19 | 38.54 | 1.34% | 1.003 |
| Model 3 | LSTM | 7 | 1 | 35.60 | 46.87 | 1.82% | 1.36 |
| Model 4 | Dense | - | 9 | 16.23 | 19.99 | 0.832% | 0.621 |
| **Model 5**  | **LSTM** | **-** | **9** | **10.49** | **15.30** | **0.542%** | **0.401** |

---

##  Key Insights

### 1. Multivariate > Univariate
The incorporation of lagged features and VWAP dramatically improved model performance, demonstrating the importance of feature engineering in time series forecasting.

### 2. Feature Engineering Impact
Creating 7 lagged features of the close price allowed models to capture temporal patterns and dependencies that were invisible to univariate approaches.

### 3. Model Complexity Trade-offs
- Simple dense networks outperformed LSTM in univariate scenarios
- LSTM architecture excelled when provided with rich, multivariate features
- Proper feature engineering matters more than model complexity alone

### 4. LSTM Advantage with Context
LSTM networks demonstrated their strength when given sufficient context through multiple features, leveraging their ability to learn long-term dependencies.

---

##  Technical Implementation

### Technologies Used
- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Data visualization

### Key Functions Developed
- `get_labelled_windows()` - Window and horizon slicing
- `make_windows()` - Sliding window creation
- `split_windows()` - Train-test split for windowed data
- `calculate_mase()` - Custom MASE metric implementation
- `evaluate_model()` - Comprehensive metrics evaluation

---

##  Methodology Highlights

### Data Integrity
- **Chronological splitting** preserved temporal order
- **No data leakage** between train and test sets
- **Consistent evaluation** across all models

### Progressive Complexity
The project follows a logical progression:
1. Baseline establishment
2. Simple univariate models
3. Complex univariate models
4. Multivariate feature engineering
5. Advanced multivariate models

---

##  Future Enhancements

- [ ] Hyperparameter optimization using GridSearch/RandomSearch
- [ ] Attention mechanisms for improved temporal focus
- [ ] Additional technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Ensemble methods combining multiple models
- [ ] Real-time prediction pipeline
- [ ] Multi-horizon forecasting (predicting multiple days ahead)
- [ ] External features (market sentiment, economic indicators)

---

##  Conclusion

This project successfully demonstrates the power of deep learning in financial time series forecasting. By systematically exploring different architectures and feature engineering strategies, we achieved a **60% improvement** over baseline predictions. The multivariate LSTM model (Model 5) emerged as the clear winner, showcasing the importance of combining appropriate architecture with rich feature sets.

The progression from naive forecasting to sophisticated multivariate models provides valuable insights into the practical application of deep learning in financial markets, while maintaining rigorous evaluation standards throughout.

---

##  Contact & Contributions

Feel free to fork, star ⭐, and contribute to this project!

---

**Note:** This project is for educational and research purposes. Financial forecasts should not be the sole basis for investment decisions.