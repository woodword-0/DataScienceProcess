import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tseries.holiday import USFederalHolidayCalendar

import matplotlib.pyplot as plt
import xgboost as xgb
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import imageio
import os
import datetime
from statsmodels.graphics.tsaplots import plot_acf
# Stats to get trend
from statsmodels.tsa.seasonal import seasonal_decompose
random.seed(42)
###############################################################################################################################################
###############################################################################################################################################
# Import data
path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/SalesQuantity.xlsx"
df = pd.read_excel(path)
df = df.sort_values('Date')
df.Date.nunique()
df = df.set_index('Date')
# df.describe()
###############################################################################################################################################
###############################################################################################################################################
# Rename Columns
# DF = df.rename(columns = {'Sales Quantity': 'ts'})
# DF = DF.resample('D').mean()
# DF.describe()
# df = df[['Sales Quantity','Date']].loc[df['Sales Quantity'] != 0]
# Rename date series 'ds' and time series 'ts'
DF = df.rename(columns = {'Date': 'ds', 'Sales Quantity': 'ts'})
# Outliers
DF = DF[(DF['ts'] < DF['ts'].mean() + 3*DF['ts'].std()) & (DF['ts'] > DF['ts'].mean() - 3*DF['ts'].std())]
# Train Test Split
DF = DF['ts']
def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()
train, test = split_data(DF, '01-Jan-2022')

plt.figure(figsize=(15,5))
plt.xlabel('time')
plt.ylabel('Sales')
plt.plot(train.index,train)
plt.plot(test.index,test)
plt.show()
DF.index
# df  =DF.copy()
# Create features
def create_features(df):
    """
    Creates time series features from datetime index
    """
    first = str(df.index[0])
    last = str(df.index[-1])
    cal = USFederalHolidayCalendar()
    hol = cal.holidays(start=first, end=last)
    df['date'] = df.index
    df['hour'] = df['date'].hour
    df['dayofweek'] = df['date'].dayofweek
    df['quarter'] = df['date'].quarter
    df['month'] = df['date'].month
    df['year'] = df['date'].year
    df['dayofyear'] = df['date'].dayofyear
    df['dayofmonth'] = df['date'].day
    # df['weekofyear'] = df['date'].dt.weekofyear
    # df['isholiday'] = np.where(
    #     df.index.to_period('D').astype('datetime64[ns]').isin(hol),
    #     1, 0
    #     )
    X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear']]
    return X
# Test train split
X_train, y_train = create_features(train), train
X_test, y_test   = create_features(test), test

X_train.shape, y_train.shape
 # Model
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
    verbose=False) # Change verbose to True if you want to see it train
X_test_pred = reg.predict(X_test)
