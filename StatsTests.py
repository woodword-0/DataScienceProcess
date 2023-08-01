import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf
# import pmdarima as pm

# Import Data
path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/SalesQuantity.xlsx"
df = pd.read_excel(path)
df.dtypes
df.isna().sum()
df = df.sort_values('Date')
df = df.set_index('Date')

# Data Preprocessing

DF = pd.get_dummies(df)
DF = DF.replace({True: 1, False: 0})
DF.dtypes
DF.corr()

heatmap
import numpy as np 
from pandas import DataFrame
import seaborn as sns
DF.columns

sns.heatmap(DF[['Sales Quantity','Order Type_Journal']])
plt.show()
DF['Sales Quantity'].plot()
ts = DF['Sales Quantity']
# Resampling
ts_month_avg = df['Sales Quantity'].resample('MS').mean()
ts_month_avg.dropna(inplace=True)
#Seasonality plot
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(ts, model='additive',period=30)
fig = decomposition.plot()
plt.show()

# plot_series = df[df.index<pd.Timestamp(df.index[1].date())]
# plot_series = ts[ts.index<pd.Timestamp(ts.index[1].date())]

plot_acf(ts_month_avg)
plt.show()
# plot_series.shape
# Tests for Stationarity
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

# print(adf_test(ts_month_avg))
# If p value is less than the critical value then there is stationarity (reject the null)
print(adf_test(ts_month_avg))