#!/usr/bin/env python
# coding: utf-8

# In[26]:


import yfinance as yf
import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from pandas.plotting import autocorrelation_plot
import numpy as np
import yfinance as yf
from scipy import stats
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
from statsmodels.tsa.api import VAR
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import BDay 
import warnings


# In[27]:


df = yf.download("AMD", start=None, end="2024-04-26", interval="1mo")

amd_pct_change = df["Adj Close"].pct_change().dropna()
amd_log_return = np.log(df["Adj Close"] / df["Adj Close"].shift(1)).dropna()
amd_log_return_skewness = amd_log_return.skew()
amd_log_return_kurtosis = amd_log_return.kurtosis()

ema_window = 25
ema_min_periods = 10 
amd_avg_abs_mean = amd_log_return.abs().rolling(window=ema_window, min_periods=ema_min_periods).mean().mean()

amd_stats = {
    "Percentage Change Mean": amd_pct_change.mean(),
    "Percentage Change Standard Deviation": amd_pct_change.std(),
    "Log Returns Mean": amd_log_return.mean(), 
    "Log Returns Standard Deviation": amd_log_return.std(), 
    "Absolute Mean of Log Returns": amd_log_return.abs().mean(), 
    "Average Absolute Mean of Log Returns": amd_avg_abs_mean,
    "Mean of Adj Close": df["Adj Close"].mean(),
    "Standard Deviation of Adj Close": df["Adj Close"].std(),
    'Skewness of Log Returns': amd_log_return_skewness,
    'Kurtosis of Log Returns': amd_log_return_kurtosis
}

print("\nAMD Statistic:")
for key, value in amd_stats.items():
    print(f"{key}: {value}")


# In[28]:


amd_max_indices = argrelextrema(amd_log_return.values, np.greater)[0]

plt.figure(figsize=(14, 8))
plt.plot(amd_log_return, color="#007acc", linewidth=1)
plt.scatter(amd_log_return.index[amd_max_indices], amd_log_return.iloc[amd_max_indices], color='red', marker='o', label='Highest Peaks')
plt.xlabel("Date", fontsize=14)
plt.ylabel("Log Percentage Return", fontsize=14)
plt.title("AMD Returns", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  
plt.tight_layout()
plt.show()


# In[29]:


amd_max_indices = argrelextrema(amd_log_return.values, np.greater)[0]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(amd_log_return.index[:-1], amd_log_return.values[:-1], label="Log Percentage Change")
axes[0].plot(amd_log_return.index[:-1], amd_log_return.abs().mean() * np.ones_like(amd_log_return[:-1]), color='green', linestyle='-', label='Absolute Mean')
axes[0].plot(amd_log_return.index[:-1], amd_log_return.abs().mean() * np.ones_like(amd_log_return[:-1]), color='blue', linestyle='--', label='Average Absolute Mean')
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Log Percentage Change")
axes[0].set_title("AMD Log Percentage Change with Absolute Mean and Average Absolute Mean")
axes[0].legend()

lag_plot(amd_log_return[:-1], lag=1, ax=axes[1], c='firebrick', s=6)
axes[1].set_title("Lag of Log Percentage Change")
axes[1].set_ylabel("Log Percentage Change (y(t+1))")
axes[1].set_xlabel("Log Percentage Change (y(t))")

plt.tight_layout()
plt.show()


# In[30]:


result_amd = adfuller(amd_log_return)
print("ADF Statistic for AMD:", result_amd[0])
print("p-value for AMD:", result_amd[1])
print("Critical Values for AMD:")
for key, value in result_amd[4].items():
    print("\t%s: %.3f" % (key, value))


# In[31]:


def ggacf(y, lag=12, plot_zero="no", alpha=0.05):
    T = len(y)
    
    y_acf = y.autocorr(lag=lag)
    print("Autocorrelation value:", y_acf) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(lag+1), [y.autocorr(lag=lag) for lag in range(lag+1)], color="orange")
    ax.axhline(y=stats.norm.ppf(1-alpha/2) / np.sqrt(T), color="steelblue", linestyle="dashed")
    ax.axhline(y=stats.norm.ppf(alpha/2) / np.sqrt(T), color="steelblue", linestyle="dashed")
    ax.axhline(y=0, color="steelblue")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title("ACF with 95% CI")
    plt.show()

ggacf(amd_log_return, lag=12)


# In[32]:


ma_square_returns = amd_log_return ** 2
ma = ma_square_returns.rolling(window=25).mean()

ema_square_returns = amd_log_return ** 2
ema = ema_square_returns.ewm(alpha=0.06, adjust=False).mean()

plt.figure(figsize=(10, 6))

plt.plot(ma, label='MA (Window = 25 days)', color='green')
plt.plot(ema, label='EMA (Smoothing Parameter = 0.06)', color='blue')

plt.title('Moving Average and Exponential Moving Average of Square Returns')
plt.xlabel('Years')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.show()


# In[33]:


amd_log_return_rescaled = 10 * amd_log_return
returns = amd_log_return_rescaled
returns = returns.dropna()
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
model = arch_model(returns, vol="Garch", p=1, q=1)
res = model.fit()

print(res.params)
print(res.summary())


# In[34]:


rolling_predictions = []
test_size = 471

for i in range(test_size):
    train = amd_log_return_rescaled[:-(test_size-i)]
    if len(train) > 100:  
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(disp="off")
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    else:
        rolling_predictions.append(np.nan)


# In[35]:


rolling_predictions = pd.Series(rolling_predictions, index=amd_log_return_rescaled.index[-471:])


# In[36]:


plt.figure(figsize=(14, 6))
true, = plt.plot(amd_log_return_rescaled.index[-354:], amd_log_return_rescaled[-354:], color="blue", linestyle="-", marker="o", markersize=5)
preds, = plt.plot(rolling_predictions.index, rolling_predictions, color="red", linestyle="--")
plt.title("Volatility Prediction for AMD - Rolling Forecast", fontsize=20)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Volatility", fontsize=16)
plt.legend(["True Returns", "Predicted Volatility"], fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[37]:


start_date = pd.Timestamp('2024-04-26')

future_dates = [start_date + BDay(i) for i in range(1, 8)]

pred = model_fit.forecast(horizon=7)

pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title("Volatility Prediction - Next 7 Trading Days", fontsize=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Volatility", fontsize=14)
plt.grid(True)

plt.xticks(future_dates, [date.strftime("%m-%d-%Y") for date in future_dates], rotation=45)

plt.tight_layout()
plt.show()


# In[39]:


start_date = amd_log_return_rescaled.index[-1] + BDay(1)

future_dates = [start_date + BDay(i) for i in range(1, 64)]

pred = model_fit.forecast(horizon=63)

pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title("Volatility Prediction - Next 63 Trading Days", fontsize=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Volatility", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[42]:


future_dates = [start_date + BDay(i) for i in range(1, 253)]

pred = model_fit.forecast(horizon=252)

pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title("Volatility Prediction - Next 252 Trading Days", fontsize=20)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Volatility", fontsize=14)
plt.grid(True)
plt.show()


# In[ ]:




