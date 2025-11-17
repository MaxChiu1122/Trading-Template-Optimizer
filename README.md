# Trading Strategy Optimizer

A comprehensive Python-based trading strategy backtester and optimizer that uses Excel for configuration and provides robust performance analysis with visualization capabilities. This system enables traders and quantitative analysts to design, test, and optimize trading strategies through an intuitive Excel interface.

## 🚀 Features

### Core Capabilities
- **Excel-Driven Configuration**: Define strategies, indicators, and parameters directly in Excel - no coding required for strategy setup
- **Dynamic Indicator Building**: Support for TA-Lib indicators and custom arithmetic combinations with parameter support
- **Rolling Window Optimization**: Hyperopt-powered parameter optimization with train/test splits for robust strategy validation
- **Multi-Period Position Holding**: Support for both single-day (intraday) and multi-day position management
- **Cumulative Position Sizing**: Professional pyramiding strategy with position limits and opposite signal reduction
- **Comprehensive Performance Metrics**: 15+ performance metrics including Sharpe ratio, max drawdown, win rate, accuracy, and more
- **Interactive Visualizations**: Equity curves, trade markers, drawdown charts, and performance visualizations
- **Modular Architecture**: Clean, maintainable code with separated concerns for easy extension

### Advanced Features
- **Stop Loss & Take Profit**: Built-in support for stop loss and take profit orders
- **Position Management**: Professional cumulative position sizing with configurable limits
- **Multi-Objective Optimization**: Optimize multiple metrics simultaneously with custom weights
- **Time-Series Cross-Validation**: Rolling window backtesting prevents overfitting
- **Reproducible Results**: Random seed support for consistent optimization results

## 📊 Sample Results

The system generates comprehensive trading analysis including:
- **Performance Metrics**: Returns, Sharpe ratio, drawdowns, win rate, accuracy, and more
- **Trade-by-Trade Analysis**: Detailed entry/exit tracking with PnL per trade
- **Visual Equity Curves**: Price charts with buy/sell markers, equity curves, and drawdown visualization
- **Parameter Optimization Results**: Best parameters across rolling windows with train/test metrics
- **Excel Integration**: All results automatically written back to Excel for easy analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/excel-trading-optimizer.git
cd excel-trading-optimizer
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install TA-Lib

TA-Lib is required for technical analysis indicators. Installation varies by platform:

#### Windows

```bash
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl  # Adjust version for your Python version
```

#### macOS
```bash
brew install ta-lib
pip install TA-Lib
```

#### Linux
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

## 📈 Quick Start

### Basic Backtesting

1. **Open the Excel template**: `excel/trading_template.xlsx`

2. **Configure your strategy** in the "Strategy Logic Builder" sheet:
   - Define entry conditions (Enter-Buy, Enter-Sell)
   - Define exit conditions (Exit-long, Exit-short)
   - Optionally add stop loss and take profit rules

3. **Set up indicators** in the "Indicator Builder" sheet:
   - Create custom arithmetic indicators
   - Add TA-Lib indicators

4. **Set parameters** in the "Dashboard" sheet:
   - Define initial parameter values
   - Set date range for backtesting

5. **Run the backtest**:
```python
from main import main

# Run basic backtest
main(optimize=False, max_position=3)
```

Results will be written to the Excel file in the "Results" and "Visualization" sheets.

### Parameter Optimization

1. **Define parameter ranges** in the "Dashboard" sheet:
   - Set Min, Max, and Step for each parameter to optimize

2. **Configure optimization settings**:
   - Set train window size (e.g., 40 days)
   - Set test window size (e.g., 20 days)
   - Specify which parameters to optimize
   - Set optimization objective (MAX or MIN)
   - Set maximum evaluations (e.g., 100)

3. **Set objective weights**:
   - Define which metrics to optimize (Return, Sharpe, Accuracy, etc.)
   - Assign weights to each metric

4. **Run optimization**:
```python
from main import main

# Run parameter optimization
main(optimize=True, max_position=5)
```

Optimization results will be written to the "Optimization" and "Train" sheets, showing best parameters for each rolling window.

## 📁 Project Structure

```
Trading-Template-Optimizer/
├── main.py                    # Main entry point and workflow orchestration
├── strategy.py               # Strategy logic evaluation and trade generation
├── optimizer.py              # Rolling window optimization with Hyperopt
├── indicator_builder.py      # Dynamic indicator construction
├── performance_metrics.py    # Performance calculation and analysis
├── excel_io.py              # Excel file I/O operations
├── generate_visuals.py       # Visualization and plotting
├── setup.py                 # Package setup script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── POSITION_MANAGEMENT.md   # Position management documentation
├── excel/
│   └── trading_template.xlsx # Excel configuration template
├── data/                    # Sample data files
│   ├── data_sample.xlsx
│   └── Pt.txt
└── images/                  # Generated visualization outputs
    └── trading_visualization.png
```

## 🔧 Configuration Guide

### Strategy Logic Builder

Define your trading rules in Excel using this format:

| Rule Type | Column A | Operator | Column B/Value | Action at | Logic Type | Position |
|-----------|----------|----------|----------------|-----------|------------|----------|
| Enter-Buy | Open | < | Final_Price | Open | END | BP |
| Enter-Sell | Open | > | Final_Price | Open | END | SP |
| Exit-long | Close | == | Close | Close | END | - |
| Exit-short | Close | == | Close | Close | END | - |
| StopLoss-long | Low | < | EntryPrice - 10 | Close | END | - |
| TakeProfit-long | High | > | EntryPrice + 20 | Close | END | - |

**Operators**: `<`, `>`, `<=`, `>=`, `==`, `!=`  
**Logic Types**: `AND`, `OR`, `END` (for chaining conditions)  
**Action at**: Price field to use for entry/exit (Open, High, Low, Close, or custom indicator)

### Indicator Builder

Create custom indicators with parameter support using arithmetic operations:

| Indicator Name | Indicator A | Operator | Value / Param | Combination |
|----------------|-------------|----------|---------------|-------------|
| Final_Price | Oy | * | OP | + |
| Final_Price | Hy | * | HP | + |
| Final_Price | Ly | * | LP | + |
| Final_Price | Cy | * | CP | + |
| Final_Price | Open | * | OtP | + |
| Final_Price | Pt | * | PP | END |

**Operators**: `+`, `-`, `*`, `/`, `**` (power)  
**Combination**: `+`, `-`, `*`, `/`, `END` (for chaining operations)

### TA-Lib Indicators

Create indicators using TA-Lib functions:

| TA-Lib Name | TA-Lib Function | In order Indicators | In order Param |
|-------------|-----------------|---------------------|----------------|
| ADX | ADX | High, Low, Close | ADX |
| RSI | RSI | Close | RSI |
| ShortEnter, middle, LongEnter | BBANDS | Close | Period, Enter, Enter |

**Multiple outputs**: Use comma-separated names for functions that return multiple values (e.g., BBANDS)

### Parameter Configuration

Set parameters in the Dashboard sheet:

| Parameter | Initial | Min | Max | Step |
|-----------|---------|-----|-----|------|
| BP | 1 | 1 | 5 | 1 |
| SP | 1 | 1 | 5 | 1 |
| OP | 0.2 | 0.1 | 0.5 | 0.05 |
| HP | 0.2 | 0.1 | 0.5 | 0.05 |
| LP | 0.2 | 0.1 | 0.5 | 0.05 |
| CP | 0.2 | 0.1 | 0.5 | 0.05 |
| OtP | 0.1 | 0.05 | 0.2 | 0.01 |
| PP | 0.1 | 0.05 | 0.2 | 0.01 |

**Initial**: Starting value for backtesting  
**Min/Max/Step**: Range for optimization (leave empty to skip optimization)

### Optimization Settings

Configure rolling window optimization:

| Settings | Value |
|----------|-------|
| Train Window Size (days) | 40 |
| Test Window Size (days) | 20 |
| Optimize Parameters | BP, SP, OP, HP, LP, CP, OtP, PP |
| Objective Type | MAX |
| Max Evaluation | 100 |
| Random Seed | 42 |

**Objective Weights**:

| Optimization Metric | Weight |
|---------------------|--------|
| AccReturn | 1 |
| Sharpe | 0 |
| Max Drawdown | 0 |
| Accuracy | 2 |
| SqrtMSE [%] | 0 |

Set weights to 0 to exclude metrics from optimization.

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- **Return [%]**: Total return percentage
- **PnL**: Profit and Loss in dollars
- **Equity Start/Final [$]**: Starting and ending equity

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown [%]**: Maximum peak-to-trough decline
- **SqrtMSE [%]**: Root mean square error for prediction accuracy

### Trade Statistics
- **# Trades**: Total number of trades
- **Win Rate [%]**: Percentage of profitable trades
- **Avg. Trade [%]**: Average return per trade

### Prediction Metrics
- **Accuracy [%]**: Percentage of correct price direction predictions
- **SqrtMSE [%]**: Prediction error metric

## 🎯 Strategy Types Supported

The system supports various trading strategies:

- **Trend Following**: Moving average crossovers, momentum strategies, breakout systems
- **Mean Reversion**: RSI, Bollinger Bands, oversold/overbought conditions
- **Breakout**: Price level breaks, volatility breakouts, pivot point breakouts
- **Multi-Factor**: Complex combinations of technical indicators
- **Machine Learning**: Integration with prediction models (Final_Price predictions)

## 💼 Position Management

This system implements **professional cumulative position sizing** following institutional trading practices:

### Key Features

- **Pyramiding**: Multiple buy signals add up (e.g., IBS Reversion + Gap and Go = 2×BP position)
- **Position Reduction**: Opposite signals reduce position rather than reversing (e.g., +2 long + sell signal = +1 long)
- **Risk Limits**: Configurable maximum position (default ±3) prevents over-exposure
- **Flexible Sizing**: Each strategy contributes independently to total position

### Example Scenarios

| Current Position | Signal | New Position | Explanation |
|------------------|--------|--------------|-------------|
| 0 (flat) | Buy (BP=1) | +1 | Open long |
| +1 | Buy (BP=1) | +2 | Add to position (pyramiding) |
| +2 | Sell (SP=1) | +1 | Reduce position (take profit) |
| +1 | Sell (SP=1) | 0 | Close position |
| 0 | Sell (SP=1) | -1 | Open short |
| -2 | Buy (BP=1) | -1 | Cover short (reduce) |

📖 **See [POSITION_MANAGEMENT.md](POSITION_MANAGEMENT.md) for detailed documentation**

## 🔄 Optimization Features

### Rolling Window Backtesting
- **Time-Series Cross-Validation**: Prevents look-ahead bias
- **Train/Test Split**: Separate training and testing periods for each window
- **Window Overlap**: Configurable step size between windows

### Hyperparameter Optimization
- **Bayesian Optimization**: Uses Hyperopt's TPE algorithm for efficient search
- **Multiple Objectives**: Optimize multiple metrics with custom weights
- **Reproducible**: Random seed support for consistent results

### Robust Validation
- **Separate Train/Test Metrics**: Evaluate performance on unseen data
- **Multiple Windows**: Test strategy across different market conditions
- **Best Parameter Selection**: Automatically selects best parameters based on objective

## 📝 Usage Examples

### Example 1: Simple Moving Average Crossover

```python
# In Excel Strategy Logic Builder:
# Enter-Buy: SMA_20 > SMA_50
# Exit-long: SMA_20 < SMA_50
# Position: BP=1

from main import main
main(optimize=False, max_position=1)
```

### Example 2: RSI Mean Reversion with Optimization

```python
# In Excel:
# - Set RSI indicator (period=14)
# - Enter-Buy: RSI < 30
# - Exit-long: RSI > 70
# - Optimize RSI period: 10-20, step=1

from main import main
main(optimize=True, max_position=2)
```

### Example 3: Multi-Strategy Portfolio

```python
# Combine multiple strategies with cumulative positions
# - IBS Reversion (BP=1)
# - Gap and Go (BP=1)
# - Noise Area Breakout (SP=1)
# Max position: 3

from main import main
main(optimize=True, max_position=3)
```

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib Import Error**
   - Ensure TA-Lib C library is installed before installing Python package
   - On Windows, use pre-compiled wheel files

2. **Excel File Locked**
   - Close Excel before running the script
   - Ensure no other processes are accessing the file

3. **Missing Columns Error**
   - Verify all columns referenced in Strategy Logic exist in market data
   - Check Indicator Builder for correct column names

4. **Optimization Takes Too Long**
   - Reduce max_evals parameter
   - Reduce train/test window sizes
   - Optimize fewer parameters

5. **No Trades Generated**
   - Check entry conditions are not too restrictive
   - Verify market data date range
   - Check indicator calculations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical Analysis Library
- [Hyperopt](https://github.com/hyperopt/hyperopt) - Hyperparameter Optimization
- [Pandas](https://github.com/pandas-dev/pandas) - Data Analysis Library
- [OpenPyXL](https://openpyxl.readthedocs.io/) - Excel file manipulation

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: 2025  
**Python Version**: 3.8+
