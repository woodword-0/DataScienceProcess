# Import packages
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
import pyaf.ForecastEngine as autof
import pickle
from math import floor
from prophet import Prophet
import os 

# Import packages
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import KFold

import plotly.express as px
np.random.seed(9001)

# Import data
path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/InvoiceSet1.xlsx"
df = pd.read_excel(path)
df.isna().sum()
df.dropna(inplace=True)
df['Time'] = pd.to_datetime(df['Time'])
df.sort_values('Time',inplace=True)
df.set_index(df.Time, inplace=True) 
# Outliers
df = df[(df['Sales Quantity'] < df['Sales Quantity'].mean() + 3*df['Sales Quantity'].std()) & (df['Sales Quantity'] > df['Sales Quantity'].mean() - 3*df['Sales Quantity'].std())]

dfts = df[['Sales Amount','Sales Quantity']].resample('W').sum().fillna(0)
dfts.shape
dfts.index
dfts.to_csv('SalesQuantityAgg.csv')
dfts.isna().sum()
df.columns
os.cwd()
# Create exogeneous dataset
df.dtypes
df.columns
df.Time
dft = df.groupby('Time').sum()
dft
# def create_features(df):
"""
Creates time series features from datetime index
"""
first = df.Time.iloc[0]
last = df.Time.iloc[-1]
cal = USFederalHolidayCalendar()
hol = cal.holidays(start=first, end=last)
# df['hour'] = df['Time'].dt.hour
df['dayofweek'] = df['Time'].dt.dayofweek
df['quarter'] = df['Time'].dt.quarter
df['month'] = df['Time'].dt.month
df['year'] = df['Time'].dt.year
df['dayofyear'] = df['Time'].dt.dayofyear
df['dayofmonth'] = df['Time'].dt.day
# df['weekofyear'] = df['Time'].dt.weekofyear
df['isholiday'] = np.where(
    df['Time'].isin(hol),
    1, 0
    )
ts_cv = TimeSeriesSplit(
    n_splits=3,
    gap=30,
)

# from sklearn.compose import ColumnTransformer
codes,uniques = pd.factorize(df['Product Name'])
# name and class pairs for each product
pairs = dict(list(zip(codes,df['Product Name'].values)))
df['Coded Product'] = codes
# Sku
Xi = df[df['Product Name'] == df['Product Name'].iloc[0]]
X = Xi.drop(['Sales Quantity','Time','Product Name'],axis='columns')


# Global
# X = df.drop(['Sales Quantity','Time','Product Name'],axis='columns')
# Target
y = Xi['Sales Quantity']
X.dtypes
gbrt_pipeline = make_pipeline(HistGradientBoostingRegressor(random_state=42)
).set_output(transform="pandas")
# Lets evaluate our gradient boosting model with the mean absolute error of the relative demand averaged across our 5 time-based cross-validation splits:

def evaluate(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )

# Evaluate model 
evaluate(gbrt_pipeline, X, y, cv=ts_cv)

# if __name__ == '__main__':
#    # generate a daily signal covering one year 2016 in a pandas dataframe
   N = 360
				  
   # create a forecast engine, the main object handling all the operations
   lEngine = autof.cForecastEngine()

   # get the best time series model for predicting one week
   lEngine.train(iInputDS=X, iTime='Date', iSignal='Signal', iHorizon=30);
   lEngine.getModelInfo() # => relative error 7% (MAPE)

   # predict one week
   df_forecast = lEngine.forecast(iInputDS=df_train, iHorizon=7)
   # list the columns of the forecast dataset
   print(df_forecast.columns)

   # print the real forecasts
   # Future dates : ['2017-01-19T00:00:00.000000000' '2017-01-20T00:00:00.000000000' '2017-01-21T00:00:00.000000000' '2017-01-22T00:00:00.000000000' '2017-01-23T00:00:00.000000000' '2017-01-24T00:00:00.000000000' '2017-01-25T00:00:00.000000000']
   print(df_forecast['Date'].tail(7).values)

   # signal forecast : [ 9.74934646  10.04419761  12.15136455  12.20369717  14.09607727 15.68086323  16.22296559]
   print(df_forecast['Signal_Forecast'].tail(7).values)
df.nunique()
df.columns




df.isna().sum()
# ts = df['Sales Amount']
# df.shape,df.nunique()
# df.groupby('Product Name').sum('Sales Amount').sort_values('Sales Amount')
# df.sort_values("Date",inplace=True)
df[['Year','Day']]
df.dtypes
def plot_cross_val(n_splits: int,
                   splitter_func,
                   df: pd.DataFrame,
                   title_text: str) -> None:
  
    """Function to plot the cross validation of various
    sklearn splitter objects."""

    split = 1
    plot_data = []

    for train_index, valid_index in splitter_func(n_splits=n_splits).split(df):
        plot_data.append([train_index, 'Train', f'{split}'])
        plot_data.append([valid_index, 'Test', f'{split}'])
        split += 1

    plot_df = pd.DataFrame(plot_data,
                           columns=['Index', 'Dataset', 'Split'])\
                           .explode('Index')

    fig = go.Figure()
    for _, group in plot_df.groupby('Split'):
        fig.add_trace(go.Scatter(x=group['Index'].loc[group['Dataset'] == 'Train'],
                                 y=group['Split'].loc[group['Dataset'] == 'Train'],
                                 name='Train',
                                 line=dict(color="blue", width=10)
                                 ))
        fig.add_trace(go.Scatter(x=group['Index'].loc[group['Dataset'] == 'Test'],
                                 y=group['Split'].loc[group['Dataset'] == 'Test'],
                                 name='Test',
                                 line=dict(color="goldenrod", width=10)
                                 ))

    fig.update_layout(template="simple_white", font=dict(size=20),
                      title_text=title_text, title_x=0.5, width=850,
                      height=450, xaxis_title='Index', yaxis_title='Split')

    legend_names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in legend_names) else legend_names.add(trace.name))

    return fig.show()
  
  

    


# Import packages
from sklearn.model_selection import TimeSeriesSplit

# Plot the time series cross validation splits
plot_cross_val(n_splits=5,
               splitter_func=TimeSeriesSplit,
               df=ts,
               title_text='Time Series Cross-Validation')   

ts.index
import plotly.express as px
import pandas as pd


def plot_time_series(df: pd.DataFrame) -> None:
    """General function to plot Time Series."""
    
    fig = px.line(df, x='Time', y='Sales Quantity',
                  labels={'Time': 'Date', 'Sales Quantity': 'Sales Quantity'})
                  
    fig.update_layout(template="simple_white", font=dict(size=18),
                      title_text='Sales Quantity', width=650,
                      title_x=0.5, height=400)

    return fig.show()
    
# Outliers
DF = df[(df['Sales Quantity'] < df['Sales Quantity'].mean() + 3*df['Sales Quantity'].std()) & (df['Sales Quantity'] > df['Sales Quantity'].mean() - 3*df['Sales Quantity'].std())]
# SKU
DF.nunique()
DF.loc[DF['Product Name'] == DF['Product Name'].iloc[0]]
# Groups products by variance
# DF.groupby('Product Name').agg(["mean", "median", "var"])
DF.dropna(inplace=True)
DF.isna().sum()
vardf = DF[['Product Name','Sales Amount','Sales Quantity']].groupby('Product Name').agg(["var"])
salesdf = DF[['Product Name','Sales Amount']].groupby('Product Name').sum()
voldf = DF[['Product Name','Sales Amount','Sales Quantity']].groupby('Product Name').agg(["var"])
vardf['Products'] = vardf.index
vardf[('Sales Quantity','var')]


var1df = pd.DataFrame()
var1df['Products'] = vardf.index
var1df['Sales Quantity Var'] = vardf[('Sales Quantity','var')].values
var1df['Sales Amount Var'] = vardf[('Sales Amount','var')].values
var1df
var1df.columns
var1df.isna().sum()
vardf.index
fig = px.line(var1df, x='Products', y='Sales Quantity Var',
                  labels={'Product': 'Products', 'Sales Quantity': 'Sales Quantity Var'})
                  
fig.update_layout(template="simple_white", font=dict(size=18),
                    title_text='Sales Quantity', width=650,
                    title_x=0.5, height=400)

fig.show()

df.hist()
df.plot('Sales Quantity')
plt.show()
vardf.plot()
plt.show()
DF.columns
# Plot the time series
plot_time_series(df=DF)




# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(Product Name, Sales Amount, Sales Quantity, Year, Quarter, Month, Day)
# dataset = dataset.drop_duplicates()
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import TimeSeriesSplit

# Paste or type your script code here:
dataset.dropna(inplace=True)
################################################################################################################################################################################################
# Outliers
# df = dataset.copy()
df = df[(df['Sales Quantity'] < df['Sales Quantity'].mean() + 3*df['Sales Quantity'].std()) & (df['Sales Quantity'] > df['Sales Quantity'].mean() - 3*df['Sales Quantity'].std())]
##################################################################################################################################################################################################
# Index
df.sort_values('Time',inplace=True)
df['Time'] = pd.to_datetime(df['Time'])
##################################################################################################################################################################################################
# Create exogeneous dataset
first = df.Time.iloc[0]
last = df.Time.iloc[-1]
cal = USFederalHolidayCalendar()
hol = cal.holidays(start=first, end=last)
df['dayofweek'] = df['Time'].dt.dayofweek
df['quarter'] = df['Time'].dt.quarter
df['month'] = df['Time'].dt.month
df['year'] = df['Time'].dt.year
df['dayofyear'] = df['Time'].dt.dayofyear
df['dayofmonth'] = df['Time'].dt.day
df['isholiday'] = np.where(
    df['Time'].isin(hol),
    1, 0
    )
##################################################################################################################################################################################################
# Cross-validation 
ts_cv = TimeSeriesSplit(
    n_splits=3,
    gap=30,
)
##################################################################################################################################################################################################
# Plotting
def plot_cross_val(n_splits: int,
                    gap: int,
                   splitter_func,
                   df: pd.DataFrame,
                   title_text: str) -> None:
  
    """Function to plot the cross validation of various
    sklearn splitter objects."""

    split = 1
    plot_data = []

    for train_index, valid_index in splitter_func(n_splits=n_splits,gap=gap).split(df):
        plot_data.append([train_index, 'Train', f'{split}'])
        plot_data.append([valid_index, 'Test', f'{split}'])
        split += 1

    plot_df = pd.DataFrame(plot_data,
                           columns=['Index', 'Dataset', 'Split'])\
                           .explode('Index')

    fig = go.Figure()
    for _, group in plot_df.groupby('Split'):
        fig.add_trace(go.Scatter(x=group['Index'].loc[group['Dataset'] == 'Train'],
                                 y=group['Split'].loc[group['Dataset'] == 'Train'],
                                 name='Train',
                                 line=dict(color="blue", width=10)
                                 ))
        fig.add_trace(go.Scatter(x=group['Index'].loc[group['Dataset'] == 'Test'],
                                 y=group['Split'].loc[group['Dataset'] == 'Test'],
                                 name='Test',
                                 line=dict(color="goldenrod", width=10)
                                 ))

    fig.update_layout(template="simple_white", font=dict(size=20),
                      title_text=title_text, title_x=0.5, width=850,
                      height=450, xaxis_title='Index', yaxis_title='Split')

    legend_names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in legend_names) else legend_names.add(trace.name))

    return fig.show()
##################################################################################################################################################################################################
# Plot cross-validation splits
ts = df['Sales Quantity']
plot_cross_val(n_splits=5,gap = 30,
            splitter_func=TimeSeriesSplit,
            df=ts,
            title_text='Time Series Cross-Validation')   

