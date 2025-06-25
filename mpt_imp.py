import numpy as np 
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

#on avg there are 252 trading days in a year.
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

#stocks I am taking
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

#historical dates
start_date = '2018-01-01'
end_date = '2025-05-27'

def download_data():
    stock_data = {} #ill use name of stock as key
    #stock values range will be between 2018 and 2025
    for stock in stocks:
        ticker = yf.Ticker(stock)
        # Only considering closing prices
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_returns(data):
    #NORMALIZATION - to measure all variables in comparable metric
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_covar(returns):
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weights):
    #we are after the annual returns
    portfolio_returns = np.sum(returns.mean()* weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    print("Expected Portfolio Returns(Mean):", portfolio_returns)
    print("Expected Portfolio Volatility(Standard Deviation):", portfolio_volatility)

def generate_portfolios(returns):

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.rand(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean()* w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def show_portfolios(returns, risks):
    plt.figure(figsize=(10,6))
    plt.scatter(risks, returns, c=returns/risks, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Portfolio Returns')
    plt.ylabel('Expected Volatility')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def statistics(returns, weights):
    portfolio_returns = np.sum(returns.mean()* weights) * NUM_TRADING_DAYS
    portfolio_risks = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_returns, portfolio_risks, portfolio_returns / portfolio_risks])

#scipy optimize module can find min of a given function
#max of a f(x) is the min of -f(x)
def sharpe_pf(weights, returns):
    return -statistics(returns, weights)[2]

def optimize_portfolios(weights, returns):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=sharpe_pf, x0=weights[0], args=returns, method='SLSQP', constraints=constraints, bounds=bounds)

def print_optimal_portfolio(optimum, returns):
    print("Optimal Portfolio:", optimum['x'].round(3))
    print("Expected Returns, volatility and Sharpe Ratio:", statistics(returns, optimum['x'].round(3)))

def show_optimal_portfolio(opt, ret, portfolio_ret, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_ret, c=portfolio_ret / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Portfolio Returns')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(ret, opt['x'], )[1], statistics(ret, opt['x'])[0], 'g*', markersize=20.0)
    plt.show()

if __name__ == '__main__':
    data = download_data()
    show_data(data)
    log_daily_returns = calculate_returns(data)
    #stats(log_daily_returns)
    print("The Covariance of given stocks:")
    show_covar(log_daily_returns)
    weights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)

    optimum = optimize_portfolios(weights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)