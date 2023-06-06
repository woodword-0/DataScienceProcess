# Math and data 
import tensorflow as tf
import numpy as np
import pandas as pd

# Serving
import requests
import re
import json

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Time series specific
import datetime
from pandas.plotting import autocorrelation_plot
from pandas_market_calendars import get_calendar, MarketCalendar
from datetime import datetime, timedelta

# Weather data
from geopy.geocoders import Nominatim
#Stats models
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
# Other models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
###############################################################################################################################################
###############################################################################################################################################
# Input data
path = "C:/CodeTest1/Venv1/SalesFeb2023/"
file1 = "Query1.xlsx"
file2 = "Query2.xlsx"
df1 = pd.read_excel(path + file1)
df2 = pd.read_excel(path + file2)

df = pd.read_pickle('dataframe.pkl')
df.columns
###############################################################################################################################################
###############################################################################################################################################
# Finding a specific customer
# define the search phrase
# search_phrase = 'Aappakadai Indian Chettinad - Santa Clara, CA'

# # create a regular expression pattern
# pattern = re.compile(search_phrase, flags=re.IGNORECASE)

# # create a boolean mask to filter the dataframe based on the pattern
# mask = df['CustomerName'].str.contains(pattern)

# # apply the boolean mask to the dataframe to get the filtered rows
# filtered_df = df[mask]
# # Remove unecessary columns
# filtered_df = filtered_df.drop('CustomerName',axis=1)
# # print the filtered dataframe
# print(filtered_df)
#Create Exogeneous vars
exog_df = df[['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',
                False,True,'PRCP','TMIN','TMAX','TAVG','lag_1']]
exog_df.shape


###############################################################################################################################################
###############################################################################################################################################
# Rename Columns
DF = df.rename(columns = {'C_TotalSales': 'ts'})
###############################################################################################################################################
###############################################################################################################################################
DF.columns
###############################################################################################################################################
###############################################################################################################################################
# Seasonal decompostion
decomposition = seasonal_decompose(DF['ts'], period = 30,extrapolate_trend = 4)
###############################################################################################################################################
###############################################################################################################################################
# Plot the components of the ts
Decom_Df = DF.copy()
Decom_Df.loc[:, "trend"] = decomposition.trend
Decom_Df.loc[:, "seasonal"] = decomposition.seasonal
Decom_Df.loc[:, "residual"] = decomposition.resid
###############################################################################################################################################
###############################################################################################################################################
def plot_decomposition(df, ts,trend,seasonal, residual):

  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)

  ax1.plot(df[ts], label = 'Original', c = 'b')
  ax1.legend(loc = 'best')
  ax1.tick_params(axis = 'x', rotation = 45)

  ax2.plot(df[trend], label = 'Trend', c = 'lime')
  ax2.legend(loc = 'best')
  ax2.tick_params(axis = 'x', rotation = 45)

  ax3.plot(df[seasonal],label = 'Seasonality', c = 'm')
  ax3.legend(loc = 'best')
  ax3.tick_params(axis = 'x', rotation = 45)

  ax4.plot(df[residual], label = 'Residuals', c = 'r')
  ax4.legend(loc = 'best')
  ax4.tick_params(axis = 'x', rotation = 45)
  plt.tight_layout()

  #plt.subtitle('Signal Decomposition of  %s' %(ts), x =0.5, y= 1.05, fontsize = 18)
  plt.show()
###############################################################################################################################################
###############################################################################################################################################
plot_decomposition(Decom_Df, ts = 'ts', trend = 'trend', seasonal = 'seasonal', residual = 'residual')
###############################################################################################################################################
###############################################################################################################################################
# Stationarity tests
def test_stationarity(df, ts):

  rollmean = df['ts'].rolling(window=12, center =False).mean()
  rolstd   = df['ts'].rolling(window=12, center = False).std()

  original = plt.plot(df['ts'], color='blue', label ='original')
  mean = plt.plot(rollmean, 
                  color = 'red', 
                  label = 'Rolling Mean')
  std = plt.plot(rolstd, 
                  color = 'black', 
                  label = 'Rolling Std')
  plt.legend(loc = 'best')
  plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
  plt.xticks(rotation = 45)
  plt.show(block = False)
#   plt.close()
  
  # Perform Dickey-Fuller test: Want p-value below 5%
  # Null Hypothesis (H_0): time series is not stationary
  # Alternate Hypothesis (H_1): time series is stationary
  print ('Results of Dickey-Fuller Test:')
  dftest = adfuller(df[ts], 
                    autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], 
                        index = ['Test Statistic',
                                'p-value',
                                '# Lags Used',
                                'Number of Observations Used'])
  for key, value in dftest[4].items():
      dfoutput['Critical Value (%s)'%key] = value
  print (dfoutput)
###############################################################################################################################################
###############################################################################################################################################
# Test Stationarity
test_stationarity(df = DF, ts = 'ts')
###############################################################################################################################################
###############################################################################################################################################
# Test stationarity of residuals to determine if all information is captured by model
test_stationarity(df = Decom_Df, ts = 'residual')
###############################################################################################################################################
# Run ARIMA model
def run_arima(df, ts, p, d, q):
  model = ARIMA(df[ts], exog =exog_df,order = (p, d, q))
  results_arima = model.fit()

  len_results = len(results_arima.fittedvalues)
  ts_modified = df[ts][-len_results:]

  rss = sum((results_arima.fittedvalues - ts_modified)**2)
  rmse = np.sqrt(rss/len(df[ts]))

  print("RMSE: ", rmse)

  plt.figure()
  plt.plot(df[ts])
  plt.plot(results_arima.fittedvalues, color = 'r')
  plt.show()

  return results_arima
###############################################################################################################################################
###############################################################################################################################################
model_Ar = run_arima(Decom_Df, ts = 'residual', p = 40, d = 0, q = 0)
###############################################################################################################################################
###############################################################################################################################################
pred = model_Ar.forecast(14)
pred
len(pred)
plt.figure()
plt.plot(pred)
plt.show()