# Financial Scripts Collection

A curated set of Python scripts and Jupyter notebooks for financial analysis, trading simulation, and portfolio optimization. This repository demonstrates practical implementations of quantitative finance concepts, including modern portfolio theory and reinforcement learning for trading.

---

## Contents

| File Name                  | Description                                                                                  |
|----------------------------|---------------------------------------------------------------------------------------------|
| `mpt_imp.py`               | Modern Portfolio Theory implementation for portfolio optimization using real market data.    |
| `Simple-Trading-Sim.ipynb` | Q-learning based trading environment and agent simulation in a notebook format.             |

---

## Features

### **1. Modern Portfolio Theory & Optimization (`mpt_imp.py`)**

- **Data Acquisition:** Downloads historical closing prices for selected stocks (AAPL, WMT, TSLA, GE, AMZN, DB) from Yahoo Finance (2018–2025).
- **Return Calculation:** Computes daily log returns and annualizes them.
- **Covariance & Risk Analysis:** Calculates and displays the annualized covariance matrix.
- **Portfolio Simulation:** Simulates 10,000 random portfolios with different asset allocations to estimate expected returns, volatility, and Sharpe ratios.
- **Efficient Frontier Visualization:** Plots the risk-return profiles of all simulated portfolios, color-coded by Sharpe ratio.
- **Optimization:** Uses `scipy.optimize` to maximize the Sharpe ratio under constraints (weights sum to 1, no shorting).
- **Optimal Portfolio Reporting:** Prints and visualizes the optimal portfolio allocation and its performance metrics.

**Usage:**

```python mpt_imp.py```

This will download data, run the simulation, and display plots for the efficient frontier and optimal portfolio.

---

### **2. Q-Learning Trading Simulation (`Simple-Trading-Sim.ipynb`)**

- **Custom Trading Environment:** Simulates a basic trading scenario with price, cash, and holdings as state variables.
- **Actions:** Buy, Sell, or Hold at each step.
- **Reward Function:** Based on changes in portfolio value after each action.
- **Q-Learning Agent:**
  - Discretizes state space for tractable learning.
  - Implements epsilon-greedy exploration, learning rate decay, and Q-table updates.
- **Performance Tracking:** Trains the agent over 500 episodes and plots the final portfolio value per episode.
- **Visualization:** Shows agent learning progress and trading performance over time.

**To Run:**  
Open the notebook in Jupyter or Colab and run all cells to see the simulation and learning curve.

---

## Getting Started

**Requirements:**
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `scipy`
- (For the notebook) `jupyter` or Google Colab

**Install dependencies:**

```pip install numpy pandas matplotlib yfinance scipy```

---

## Example Plots

- **Efficient Frontier:** Visualizes the trade-off between expected return and risk for thousands of portfolios.
- **Sharpe Ratio Heatmap:** Color-codes portfolios by risk-adjusted return.
- **Q-Learning Performance:** Shows improvement in trading agent's final portfolio value across episodes.

---

## Usage

- **Portfolio Optimization:** Run `mpt_imp.py` to analyze and optimize a multi-asset portfolio using historical market data.
- **Trading Simulation:** Explore `Simple-Trading-Sim.ipynb` to see how reinforcement learning can be applied to trading decisions.

---

## Contributing

Contributions, suggestions, and improvements are welcome! Please open an issue or submit a pull request.

---

## License

This project is open source and available under the MIT License.

---

## Author

*Varunkumar*  
AI & Data Science Student | *[ign-ite]*

*This repository is intended for educational and research purposes. Not financial advice.*
