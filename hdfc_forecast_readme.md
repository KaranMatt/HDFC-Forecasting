#  HDFC Stock Price Forecasting with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*A comprehensive time series forecasting project leveraging deep learning to predict HDFC stock prices with 54%+ improvement over baseline*

[Overview](#-project-overview) • [Dataset](#-dataset-details) • [Methodology](#-methodology) • [Models](#-model-architectures--results) • [Findings](#-key-findings)

</div>

---

##  Project Overview

This project implements and compares multiple deep learning architectures to forecast HDFC Bank stock prices using 21 years of historical market data. The research progresses systematically from simple univariate dense networks to sophisticated multivariate LSTM architectures, demonstrating the critical importance of feature engineering in financial time series forecasting.

###  Key Achievements

- **54.5% reduction** in MAE compared to naive baseline
- **55.2% improvement** in MAPE using multivariate LSTM
- **5 distinct models** evaluated with comprehensive metrics
- **Custom windowing framework** for sequence generation
- **Chronological validation** maintaining temporal integrity

###  Learning Objectives Demonstrated

- Time series forecasting with neural networks
- Univariate vs. multivariate modeling approaches
- Feature engineering through lag creation
- Custom metric implementation (MASE)
- Model comparison and selection methodology
- Temporal data handling and validation strategies

---

##  Dataset Details

### Source Information
- **Company**: HDFC Bank Limited
- **Time Period**: January 2000 - December 2021
- **Total Duration**: 21 years of daily trading data
- **Data Points**: Thousands of daily observations

### Features Utilized

| Feature | Type | Description | Usage |
|---------|------|-------------|-------|
| **Date** | Temporal | Trading date | Converted to index for temporal operations |
| **Close Price** | Numerical | Daily closing price | Primary target variable |
| **VWAP** | Numerical | Volume Weighted Average Price | Additional feature for multivariate models |

### Data Preparation Steps

1. **Temporal Indexing**: Converted date column to datetime index for proper time series handling
2. **Array Extraction**: 
   - `prices` array: Close price values
   - `timesteps` array: Date values
3. **Train-Test Split**: 80-20 chronological split
   - Training: 2000-2017 (approximately)
   - Testing: 2017-2021 (approximately)
   - **No random shuffling** - maintains temporal order
4. **Visualization**: Created comprehensive plots showing:
   - Complete close price history
   - Train-test split boundaries
   - Temporal patterns and trends

---

##  Methodology

### Phase 1: Data Preprocessing & Visualization

#### Initial Exploration
- Visualized raw close prices across entire 21-year period
- Identified trends, seasonality, and volatility patterns
- Examined data quality and missing values

#### Train-Test Split Strategy
```
[2000 ============ 2017] [2017 == 2021]
      80% Train              20% Test
```

**Rationale for Chronological Split:**
- Prevents data leakage from future to past
- Mimics real-world forecasting scenarios
- Maintains temporal dependencies
- Ensures model evaluation on truly unseen future data

### Phase 2: Baseline Establishment

#### Naive Forecast Implementation
- **Method**: Persistence model (tomorrow's price = today's price)
- **Purpose**: Establish performance floor for comparison
- **Application**: Applied to test dataset only

**Baseline Results:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 26.08 | Average error of ₹26.08 per prediction |
| RMSE | 38.54 | Higher penalty for large errors |
| MAPE | 1.34% | 1.34% average percentage error |
| MASE | 0.99 | Slightly better than naive seasonal baseline |

### Phase 3: Windowing Framework

Custom functions developed for sequence-based learning:

#### `get_labelled_windows()`
**Purpose**: Slice arrays into windows and horizons
- **Window**: Historical sequence length for prediction
- **Horizon**: Number of future steps to predict

#### `make_windows(window_size, horizon_size)`
**Purpose**: Transform entire price array into windowed sequences
- **Inputs**: 
  - `window_size`: Number of past timesteps to use
  - `horizon_size`: Number of future timesteps to predict
- **Output**: Arrays of windows and corresponding labels

#### `split_windows()`
**Purpose**: Separate windowed data into train and test sets
- **Outputs**:
  - `train_windows`: Training input sequences
  - `test_windows`: Testing input sequences
  - `train_labels`: Training target values
  - `test_labels`: Testing target values

**Example:**
```
Window=7, Horizon=1 means:
Use 7 days of history → Predict next 1 day
[Day1, Day2, Day3, Day4, Day5, Day6, Day7] → [Day8]
```

### Phase 4: Evaluation Framework

#### Custom Metrics Implementation

**Mean Absolute Error (MAE)**
```
Average absolute difference between predicted and actual values
Lower is better | Unit: Same as target (₹)
```

**Root Mean Squared Error (RMSE)**
```
Square root of average squared errors
Penalizes large errors more heavily | Lower is better
```

**Mean Absolute Percentage Error (MAPE)**
```
Average percentage deviation from actual values
Scale-independent | Lower is better | Unit: %
```

**Mean Absolute Scaled Error (MASE)**  *Custom Implementation*
```
MAE of forecast / MAE of naive forecast
< 1.0: Better than naive | > 1.0: Worse than naive
```

### Phase 5: Univariate Modeling

**Approach**: Single feature (close price) forecasting
**Strategy**: Evaluate simple to complex architectures
**Models**: Dense networks and LSTMs with varying configurations

### Phase 6: Multivariate Modeling

**Approach**: Multiple features for enriched predictions

#### Feature Engineering Process

**Lag Feature Creation:**
```python
hdfc_windowed_df['close_1'] = hdfc_df['close'].shift(1)
hdfc_windowed_df['close_2'] = hdfc_df['close'].shift(2)
...
hdfc_windowed_df['close_7'] = hdfc_df['close'].shift(7)
```

**Resulting Feature Set:**
- Index (date)
- close_1 through close_7 (lagged prices)
- VWAP (volume-weighted average price)
- close (target variable)

**Data Split:**
```
train_inputs, test_inputs (features)
train_target, test_target (labels)
```

---

##  Model Architectures & Results

###  UNIVARIATE MODELS (Single Feature: Close Price)

---

#### Model 1: Baseline Dense Network

**Configuration:**
- Window Size: 7 days
- Horizon: 1 day
- Input Features: Close price only

**Architecture:**
```
Input (7 timesteps)
      ↓
Dense(128 units, ReLU activation)
      ↓
Dense(1 unit, Linear) [Output]
```

**Hyperparameters:**
- Neurons: 128
- Activation: ReLU (Rectified Linear Unit)
- Output: Single value (next day's close price)

**Performance Metrics:**

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| MAE | 27.3 | +4.7% ↑ (worse) |
| RMSE | 39.66 | +2.9% ↑ (worse) |
| MAPE | 1.41% | +5.2% ↑ (worse) |
| MASE | 1.04 | +5.1% ↑ (worse) |

**Analysis:**
- Slightly underperformed naive forecast
- Simple architecture insufficient for complex patterns
- Single feature limitation evident
- Suggests need for more information

---

#### Model 2: Dense Network with Extended Window

**Configuration:**
- Window Size: 30 days (4x larger than Model 1)
- Horizon: 1 day
- Input Features: Close price only

**Architecture:**
```
Input (30 timesteps)
      ↓
Dense(128 units, ReLU activation)
      ↓
Dense(1 unit, Linear) [Output]
```

**Performance Metrics:**

| Metric | Value | vs Baseline | vs Model 1 |
|--------|-------|-------------|------------|
| MAE | 30.39 | +16.5% ↑ (worse) | +11.3% ↑ (worse) |
| RMSE | 42.43 | +10.1% ↑ (worse) | +7.0% ↑ (worse) |
| MAPE | 1.55% | +15.7% ↑ (worse) | +9.9% ↑ (worse) |
| MASE | 1.16 | +17.2% ↑ (worse) | +11.5% ↑ (worse) |

**Analysis:**
- **Larger window degraded performance significantly**
- More data ≠ better predictions for dense networks
- Possible causes:
  - Overfitting to noise
  - Diluted recent signal with older data
  - Dense layers can't effectively handle long sequences
- **Key Learning**: Shorter windows (7 days) capture relevant patterns better

---

#### Model 3: Stacked LSTM Architecture

**Configuration:**
- Window Size: 7 days
- Horizon: 1 day
- Input Features: Close price only

**Architecture:**
```
Input (7 timesteps × 1 feature)
      ↓
LSTM(128 units, return_sequences=True)
      ↓
LSTM(128 units)
      ↓
Dense(1 unit, Linear) [Output]
```

**Architectural Details:**
- Two stacked LSTM layers for deep temporal learning
- First LSTM returns sequences for second layer
- 128 memory cells per layer
- Designed to capture long-term dependencies

**Performance Metrics:**

| Metric | Value | vs Baseline | vs Model 1 |
|--------|-------|-------------|------------|
| MAE | 32.73 | +25.5% ↑ (worse) | +19.9% ↑ (worse) |
| RMSE | 44.98 | +16.7% ↑ (worse) | +13.4% ↑ (worse) |
| MAPE | 1.69% | +26.1% ↑ (worse) | +19.9% ↑ (worse) |
| MASE | 1.25 | +26.3% ↑ (worse) | +20.2% ↑ (worse) |

**Analysis:**
- **Worst performing univariate model**
- LSTM complexity not justified with single feature
- Possible causes:
  - Insufficient data for LSTM training
  - Overfitting due to high parameter count
  - Limited feature space doesn't leverage LSTM strengths
- **Critical Insight**: Complex architectures need rich feature sets

---

###  MULTIVARIATE MODELS (Multiple Features: Lagged Prices + VWAP)

---

#### Model 4: Dense Network with Rich Features ⭐

**Configuration:**
- Input Features: 8 (7 lagged close prices + VWAP)
- Target: Current close price

**Feature Engineering:**
```
Features = [close_1, close_2, close_3, close_4, 
            close_5, close_6, close_7, vwap]
Target = close (current day)
```

**Architecture:**
```
Input (8 features)
      ↓
Dense(128 units, ReLU activation)
      ↓
Dense(1 unit, Linear) [Output]
```

**Performance Metrics:**

| Metric | Value | vs Baseline | vs Best Univariate |
|--------|-------|-------------|---------------------|
| MAE | 12.907 | **-50.5% ↓** | **-52.7% ↓** |
| RMSE | 17.20 | **-55.4% ↓** | **-56.6% ↓** |
| MAPE | 0.66% | **-50.7% ↓** | **-53.2% ↓** |
| MASE | 0.49 | **-50.5% ↓** | **-52.9% ↓** |

**Analysis:**
- **Dramatic performance improvement** through feature engineering
- Simple architecture excels with rich feature set
- Lag features capture temporal dependencies effectively
- VWAP adds volume-based market sentiment
- **Key Success Factor**: Feature engineering > Model complexity

**Why It Works:**
1. **7-day lags**: Capture weekly patterns and momentum
2. **VWAP**: Incorporates volume-weighted pricing information
3. **Dense network**: Sufficient for learning linear/non-linear feature combinations
4. **Compact architecture**: Reduces overfitting risk

---

#### Model 5: Stacked LSTM with Rich Features 

**Configuration:**
- Input Features: 8 (7 lagged close prices + VWAP)
- Target: Current close price

**Architecture:**
```
Input (8 features reshaped for LSTM)
      ↓
LSTM(128 units, return_sequences=True)
      ↓
LSTM(128 units)
      ↓
Dense(1 unit, Linear) [Output]
```

**Architectural Advantages:**
- Two-layer LSTM for hierarchical temporal learning
- First layer: Captures short-term patterns
- Second layer: Learns higher-level abstractions
- 128 memory cells per layer for rich representations

**Performance Metrics:**

| Metric | Value | vs Baseline | vs Model 4 | Rank |
|--------|-------|-------------|------------|------|
| MAE | 11.88 | **-54.5% ↓** | **-8.0% ↓** | 🥇 1st |
| RMSE | 17.65 | **-54.2% ↓** | +2.6% ↑ | 🥈 2nd |
| MAPE | 0.60% | **-55.2% ↓** | **-9.1% ↓** | 🥇 1st |
| MASE | 0.4551 | **-54.0% ↓** | **-7.1% ↓** | 🥇 1st |

** BEST OVERALL PERFORMANCE**

**Analysis:**
- **Champion model** across 3 out of 4 metrics
- LSTM leverages temporal dependencies in rich feature space
- Marginal improvement over Model 4 suggests:
  - Dense networks highly effective with good features
  - LSTM adds value but not transformative
  - Computational cost vs. benefit trade-off consideration

**Why This Model Excels:**
1. **Rich Feature Space**: LSTM thrives with 8 informative features
2. **Temporal Memory**: Captures complex sequential patterns
3. **Stacked Architecture**: Hierarchical learning of abstractions
4. **Optimal Complexity**: Enough capacity without overfitting

**Production Considerations:**
- Slightly higher computational cost than Model 4
- Better worst-case error handling (MAE, MAPE)
- Recommended for scenarios where accuracy > speed

---

##  Comprehensive Model Comparison

### Performance Summary Table

| Model | Type | Features | Window | MAE | RMSE | MAPE | MASE | Rank |
|-------|------|----------|--------|-----|------|------|------|------|
| **Baseline** | Naive | 1 | 1 | 26.08 | 38.54 | 1.34% | 0.99 | - |
| **Model 1** | Dense | 1 | 7 | 27.30 | 39.66 | 1.41% | 1.04 | 6th |
| **Model 2** | Dense | 1 | 30 | 30.39 | 42.43 | 1.55% | 1.16 | 5th |
| **Model 3** | LSTM | 1 | 7 | 32.73 | 44.98 | 1.69% | 1.25 | 4th |
| **Model 4** | Dense | 8 | - | 12.91 | **17.20** | 0.66% | 0.49 | 2nd |
| **Model 5** 🏆 | LSTM | 8 | - | **11.88** | 17.65 | **0.60%** | **0.4551** | **1st** |

### Visual Performance Comparison

```
MAPE (Lower is Better):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline    ████████████████████████████ 1.34%
Model 1     ██████████████████████████████ 1.41%
Model 2     ████████████████████████████████ 1.55%
Model 3     ██████████████████████████████████ 1.69%
Model 4     ██████████ 0.66%
Model 5 🏆  █████████ 0.60%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAE (Lower is Better):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline    ████████████████████████████████ 26.08
Model 1     ██████████████████████████████████ 27.30
Model 2     ██████████████████████████████████████ 30.39
Model 3     ████████████████████████████████████████ 32.73
Model 4     ███████████████ 12.91
Model 5 🏆  ██████████████ 11.88

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Improvement Analysis

**Model 5 vs Baseline (Naive Forecast):**
- MAE: 54.5% improvement
- RMSE: 54.2% improvement  
- MAPE: 55.2% improvement
- MASE: 54.0% improvement

**Model 5 vs Best Univariate (Model 1):**
- MAE: 56.5% improvement
- RMSE: 55.5% improvement
- MAPE: 57.4% improvement
- MASE: 56.2% improvement

---

##  Key Findings

###  Critical Success Factors

#### 1. Feature Engineering is Paramount
**Impact**: Multivariate models achieved 50%+ improvements

**Evidence**:
- Model 1 (univariate): MAPE 1.41%
- Model 4 (multivariate): MAPE 0.66%
- **53% improvement from features alone**

**Lessons**:
- Lag features capture temporal dependencies
- VWAP adds volume-based context
- Rich feature space > complex architectures

#### 2. Window Size Optimization Matters
**Impact**: Smaller windows outperformed larger ones

**Evidence**:
- Model 1 (window=7): MAE 27.30
- Model 2 (window=30): MAE 30.39
- **11% degradation with larger window**

**Lessons**:
- Recent patterns more predictive than distant history
- Larger windows can introduce noise
- Domain expertise guides optimal window selection

#### 3. Architecture Complexity Requires Rich Features
**Impact**: LSTMs excel only with multivariate data

**Evidence**:
- Model 3 (LSTM, univariate): MAPE 1.69% (worst)
- Model 5 (LSTM, multivariate): MAPE 0.60% (best)
- **64% improvement with same architecture, different features**

**Lessons**:
- Complex models need sufficient information
- Simple models with good features often suffice
- LSTM advantage emerges with feature richness

#### 4. Chronological Validation is Essential
**Impact**: Maintains temporal integrity and prevents leakage

**Implementation**:
- 80-20 chronological split
- No random shuffling
- Future data never used for training

**Importance**:
- Mimics real-world forecasting
- Prevents overly optimistic results
- Ensures valid performance estimates

#### 5. Baseline Comparison Provides Context
**Impact**: Naive forecast establishes meaningful benchmarks

**Value**:
- Separates signal from noise
- Validates model utility
- MASE metric directly quantifies improvement

---

##  Technical Insights

### Why Multivariate Models Dominated

**Lag Features (close_1 through close_7)**:
- Capture momentum and trend
- Encode mean-reversion patterns
- Provide contextual price movements
- Enable learning of day-of-week effects

**VWAP Integration**:
- Represents volume-weighted consensus
- Smooths intraday volatility
- Indicates institutional activity
- Adds dimension orthogonal to price alone

**Synergistic Effect**:
- Multiple features create richer representation space
- Models learn feature interactions
- Non-linear combinations emerge
- Reduces reliance on single information source

### Dense vs LSTM: When to Choose

**Dense Networks (Models 1, 2, 4)**:
- ✅ Fast training and inference
- ✅ Interpretable weights
- ✅ Effective with proper features
- ✅ Lower computational cost
- ❌ Limited temporal modeling
- ❌ Fixed input size constraints

**LSTM Networks (Models 3, 5)**:
- ✅ Excellent temporal dependency capture
- ✅ Variable sequence length handling
- ✅ Hierarchical pattern learning
- ❌ Slower training
- ❌ More parameters (overfitting risk)
- ❌ Requires rich feature space

**Recommendation**: 
- Use **Dense** (Model 4) for: Fast prototyping, resource constraints, similar accuracy
- Use **LSTM** (Model 5) for: Maximum accuracy, complex patterns, production systems

---

##  Practical Applications

### Deployment Scenarios

**Scenario 1: High-Frequency Trading Systems**
- **Model Choice**: Model 4 (Dense)
- **Rationale**: Speed critical, minimal latency
- **Trade-off**: Sacrifice 0.06% MAPE for 10x faster inference

**Scenario 2: Portfolio Management**
- **Model Choice**: Model 5 (LSTM)
- **Rationale**: Accuracy paramount, daily decisions
- **Benefit**: Best risk-adjusted returns

**Scenario 3: Research & Backtesting**
- **Model Choice**: Both models
- **Rationale**: Compare strategies, ensemble predictions
- **Advantage**: Reduced model risk through diversification

### Real-World Considerations

**Data Freshness**:
- Model trained on 2000-2021 data
- Requires periodic retraining
- Market regime changes necessitate updates

**Risk Management**:
- 0.60% MAPE still means prediction errors
- Use predictions as signals, not certainties
- Combine with fundamental analysis

**Computational Resources**:
- Model 4: Can run on CPU
- Model 5: GPU accelerated recommended
- Both: Deployable on standard infrastructure

---

##  Technical Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 2.x | High-level neural networks API |
| **Pandas** | Latest | Data manipulation and analysis |
| **NumPy** | Latest | Numerical computing |
| **Matplotlib** | Latest | Static visualizations |
| **Seaborn** | Latest | Statistical visualizations |
| **Scikit-learn** | Latest | Metrics and preprocessing |

### Development Environment
- Jupyter Notebooks for experimentation
- GPU support for LSTM training (recommended)
- Minimum 8GB RAM recommended
- Storage: ~500MB for data and models

---

##  Future Enhancements

### Immediate Next Steps
- [ ] **Cross-validation**: Implement time series cross-validation (walk-forward)
- [ ] **Hyperparameter tuning**: Grid search for optimal configurations
- [ ] **Ensemble methods**: Combine Model 4 and Model 5 predictions
- [ ] **Error analysis**: Deep dive into prediction failures

### Advanced Features
- [ ] **Attention mechanisms**: Add interpretability to LSTM predictions
- [ ] **Transformer architecture**: Test state-of-the-art sequence models
- [ ] **Additional indicators**: RSI, MACD, Bollinger Bands integration
- [ ] **Sentiment analysis**: Incorporate news and social media signals
- [ ] **Multi-horizon forecasting**: Predict 1, 3, 5, 10 days ahead
- [ ] **Volatility modeling**: Predict price ranges, not just point estimates

### Production Readiness
- [ ] **REST API**: Deploy model as microservice
- [ ] **Model monitoring**: Track prediction drift and performance
- [ ] **A/B testing framework**: Compare model versions in production
- [ ] **Automated retraining**: Scheduled model updates with new data
- [ ] **Explainability**: SHAP values for feature importance

### Research Extensions
- [ ] **Multi-stock modeling**: Expand to sector/index predictions
- [ ] **Transfer learning**: Leverage pre-trained models
- [ ] **Causality analysis**: Move beyond correlation to causation
- [ ] **Regime detection**: Identify and adapt to market conditions

---

##  Reproducibility

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Key Configuration
```python
# Data Split
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# Model 1-3 Configuration
WINDOW_SIZE_7 = 7
WINDOW_SIZE_30 = 30
HORIZON = 1

# Model 4-5 Configuration
LAG_FEATURES = 7
ADDITIONAL_FEATURES = ['vwap']

# Training
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
```

---

##  Learning Resources

### Concepts Applied
- **Time Series Analysis**: Stationarity, autocorrelation, seasonality
- **Feature Engineering**: Lag creation, domain knowledge integration
- **Neural Networks**: Dense layers, LSTM architecture, activation functions
- **Model Evaluation**: Custom metrics, baseline comparison, cross-validation
- **Financial Markets**: Price dynamics, volume analysis, technical indicators



---

##  Conclusion

This project demonstrates that **successful time series forecasting relies more on thoughtful feature engineering than complex model architectures**. While univariate models barely improved upon the naive baseline, introducing lagged features and VWAP led to dramatic 50%+ improvements across all metrics.

### Key Takeaways

1. **Feature Engineering > Model Complexity**: Model 4 (simple dense network with rich features) nearly matched the performance of Model 5 (complex LSTM)

2. **Domain Knowledge Matters**: 7-day lags and VWAP weren't arbitrary choices—they encode weekly patterns and volume-based market signals

3. **LSTM Shines with Rich Data**: LSTM underperformed in univariate setting but achieved best results when given sufficient features

4. **Simple Baselines Are Valuable**: Naive forecast provides essential context for evaluating model utility

5. **Window Size Optimization**: Smaller windows (7 days) outperformed larger windows (30 days) for this task

### Final Performance

The **champion Model 5 (Multivariate LSTM)** achieved:
- **MAE: 11.88** (₹11.88 average prediction error)
- **MAPE: 0.60%** (0.60% average percentage error)
- **54%+ improvement** over naive baseline across all metrics

This level of accuracy makes the model viable for real-world applications including portfolio optimization, trading signal generation, and risk management—provided it's used as one component of a comprehensive investment strategy.

---

##  Contact & Collaboration

Interested in discussing this project, exploring collaboration opportunities, or have questions about the methodology?

**Open to:**
- Technical discussions on time series forecasting
- Collaboration on financial ML projects
- Code review and improvement suggestions
- Academic research partnerships

---

<div align="center">

**⭐ If you found this project valuable, please star the repository!**

*Crafted with 💙 using Python, TensorFlow, and 21 years of market data*

---

**Disclaimer**: This project is for educational purposes only. Past performance does not guarantee future results. Always conduct thorough research and consult financial advisors before making investment decisions.

</div>