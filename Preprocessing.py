import requests
import numpy as np
import re
import pandas as pd
import json
from pandas_market_calendars import get_calendar, MarketCalendar
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import os
import pickle
#####################################################################################################################################################################
#####################################################################################################################################################################
# Import Data
path = "C:/CodeTest1/Venv1/SalesFeb2023/"
# Customer Sales Query
file1 = "Query1.xlsx"
# Customer Postal Codes and tokens
file3 = "Customer_Token_PostalCodes.xlsx"
df1 = pd.read_excel(path + file1)
df3 = pd.read_excel(path + file3)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Sales DataFrame and customer tokens
df = pd.merge(df1, df3, on='CustomerName', how='left')
#####################################################################################################################################################################
#####################################################################################################################################################################
df = df.loc[df.token == '5D21088F-6603-4ECD-A4DA-237DE018C8FD']
df.loc[df.PostalCode == 95051].token.nunique()
#####################################################################################################################################################################
#####################################################################################################################################################################
# Add on columns for day of week and is or is not holiday
df_ts = df.copy()
# Add a column for the day of the week
df_ts['day_of_week'] = df_ts['SalesDate'].dt.strftime('%A')
# Create a calendar object for the NYSE
nyse = get_calendar('NYSE')
# Get the holiday calendar for the NYSE
nyse_holidays = nyse.schedule(start_date=df_ts['SalesDate'].min(), end_date=df_ts['SalesDate'].max())
# Add a column for whether each date is a holiday
df_ts['is_holiday'] = df_ts['SalesDate'].isin(nyse_holidays.index)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Adding weather data
# Directory for stations.txt file along with postal code
api_key = 'sLyRxkcsmqBlZUQtwoVxrIGLkTAbTosu'  # Replace with your NOAA API key
data_directory = "C:/CodeTest1/Venv1/WeatherData/ghcnddata/"
postal_code = df.iloc[1].PostalCode                                              
#####################################################################################################################################################################
#####################################################################################################################################################################
# Find 5 nearest stations using weather data textfile
def nearest_stations(postal_code,data_directory):
    geolocator = Nominatim(user_agent="myGeocoder")
    location = geolocator.geocode(f'{postal_code},USA')
    latitude, longitude = location.latitude, location.longitude
    # Find the nearest station IDs 
    columns = ["ID", "LATITUDE", "LONGITUDE", "ELEVATION", "STATE", "NAME", "GSN_FLAG", "HCN_CRN_FLAG", "WMO_ID"]
    widths = [11, 9, 10, 7, 3, 31, 4, 4, 6]
    df_stations = pd.read_fwf(os.path.join(data_directory, "ghcnd-stations.txt"), names=columns, header=None, widths=widths)
    df_stations["DISTANCE"] = ((df_stations["LATITUDE"] - latitude) ** 2 + (df_stations["LONGITUDE"] - longitude) ** 2) ** 0.5
    # Sort the DataFrame by distance and select the top 5 nearest stations
    nearest_stations = df_stations.nsmallest(5, "DISTANCE")
    nearest_stations = nearest_stations.ID.tolist()
    return nearest_stations
#####################################################################################################################################################################
#####################################################################################################################################################################
# Top 5 nearest stations
nearest_stations = nearest_stations(postal_code,data_directory)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Precipitation
startdate=df.SalesDate.min().strftime('%Y-%m-%d')
enddate=df.SalesDate.max().strftime('%Y-%m-%d')
def get_weather_data(station_id, api_key,startdate,enddate,type):
    endpoint = f'https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=GHCND:{station_id}&startdate={startdate}&enddate={enddate}&datatypeid={type}&limit=500'
    response = requests.get(endpoint, headers={'token': api_key})
    response.raise_for_status()
    data = response.json()
    return data
#####################################################################################################################################################################
#####################################################################################################################################################################
# Preciptitation DataFrame

def is_complete_dataframe(df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return len(df) == len(date_range) and (df.index == date_range).all()

weather_data = {}
nearest_station_index = 0

while not weather_data and nearest_station_index < len(nearest_stations):
    station_id = nearest_stations[nearest_station_index]
    data = get_weather_data(station_id, api_key, startdate, enddate, 'PRCP')
    
    if 'results' in data:
        weather_data = data['results']
        dfPRCP = pd.DataFrame(weather_data)
        dfPRCP.set_index(pd.to_datetime(dfPRCP['date']), inplace=True)
        
        if not is_complete_dataframe(dfPRCP, startdate, enddate):
            weather_data = {}
    
    nearest_station_index += 1

if weather_data:
    print(dfPRCP)
else:
    print("No full dataset found for the top 5 nearest stations.")

#####################################################################################################################################################################
#####################################################################################################################################################################
# Converts to degrees Celsius
dfPRCP['value'] = dfPRCP['value']/10
#####################################################################################################################################################################
#####################################################################################################################################################################
# Dataset Prep for precipitation dataframe
dfPRCP_prep = dfPRCP[['date','value']].copy()
dfPRCP_prep.rename(columns={'date': 'SalesDate', 'value': 'PRCP'}, inplace=True)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Min Temp DataFrame
def is_complete_dataframe(df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return len(df) == len(date_range) and (df.index == date_range).all()

weather_data = {}
nearest_station_index = 0

while not weather_data and nearest_station_index < len(nearest_stations):
    station_id = nearest_stations[nearest_station_index]
    data = get_weather_data(station_id, api_key, startdate, enddate, 'TMIN')
    
    if 'results' in data:
        weather_data = data['results']
        dfTMIN = pd.DataFrame(weather_data)
        dfTMIN.set_index(pd.to_datetime(dfTMIN['date']), inplace=True)
        
        if not is_complete_dataframe(dfTMIN, startdate, enddate):
            weather_data = {}
    
    nearest_station_index += 1

if weather_data:
    print(dfTMIN)
else:
    print("No full dataset found for the top 5 nearest stations.")

#####################################################################################################################################################################
#####################################################################################################################################################################
#Converts to degrees Celsius
dfTMIN['value'] = dfTMIN['value']/10
#####################################################################################################################################################################
#####################################################################################################################################################################
# Rename Columns
dfTMIN_prep = dfTMIN[['date','value']].copy()
dfTMIN_prep.rename(columns={'date': 'SalesDate', 'value': 'TMIN'}, inplace=True)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Max Temp DataFrame
def is_complete_dataframe(df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return len(df) == len(date_range) and (df.index == date_range).all()

weather_data = {}
nearest_station_index = 0

while not weather_data and nearest_station_index < len(nearest_stations):
    station_id = nearest_stations[nearest_station_index]
    data = get_weather_data(station_id, api_key, startdate, enddate, 'TMAX')
    
    if 'results' in data:
        weather_data = data['results']
        dfTMAX = pd.DataFrame(weather_data)
        dfTMAX.set_index(pd.to_datetime(dfTMAX['date']), inplace=True)
        
        if not is_complete_dataframe(dfTMAX, startdate, enddate):
            weather_data = {}
    
    nearest_station_index += 1

if weather_data:
    print(dfTMAX)
else:
    print("No full dataset found for the top 5 nearest stations.")
    
#####################################################################################################################################################################
#####################################################################################################################################################################
#Converts to degrees Celsius
dfTMAX['value'] = dfTMAX['value']/10
#####################################################################################################################################################################
#####################################################################################################################################################################
dfTMAX_prep = dfTMAX[['date','value']].copy()
dfTMAX_prep.rename(columns={'date': 'SalesDate', 'value': 'TMAX'}, inplace=True)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Create Average Daily Temperature Dataset
dfTAVG_prep = dfTMAX_prep.copy()
dfTAVG_prep['TAVG'] = (dfTMAX_prep.TMAX + dfTMIN_prep.TMIN)/2
dfTAVG_prep.drop('TMAX',axis=1,inplace=True)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Create New DataFrame
# Set time index
df_ts.set_index('SalesDate', inplace=True)
dfPRCP_prep.set_index('SalesDate', inplace=True)
dfTMIN_prep.set_index('SalesDate', inplace=True)
dfTMAX_prep.set_index('SalesDate', inplace=True)
dfTAVG_prep.set_index('SalesDate', inplace=True)

# Convert the index to a Datetime Index
dfPRCP_prep.index = pd.to_datetime(dfPRCP_prep.index)
dfTMIN_prep.index = pd.to_datetime(dfTMIN_prep.index)
dfTMAX_prep.index = pd.to_datetime(dfTMAX_prep.index)
dfTAVG_prep.index = pd.to_datetime(dfTAVG_prep.index)
# Change categorical variables
dummies1 = pd.get_dummies(df_ts.day_of_week)
dummies2 = pd.get_dummies(df_ts.is_holiday)
# Concatenate the dummies to original dataframe
df_ts = pd.concat([df_ts, dummies1, dummies2], axis='columns')
# drop the values
df_ts.drop(['day_of_week','is_holiday','CustomerName','token','PostalCode'], axis='columns',inplace=True)
#####################################################################################################################################################################
#####################################################################################################################################################################
# New DataFrame
merged_df = pd.concat([df_ts, dfPRCP_prep,dfTMIN_prep, dfTMAX_prep, dfTAVG_prep], axis=1)
#####################################################################################################################################################################
#####################################################################################################################################################################
# Find Optimal lag
#Create Exogeneous vars
exog_df = merged_df[['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',False,True,'PRCP','TAVG']]
#####################################################################################################################################################################
#####################################################################################################################################################################
# BIC Criterion to compute optimal lag 
# Restric to Daily Total Sales Column
# load dataset
df = merged_df.C_TotalSales.copy()
# Define a range of lag lengths to consider
lags = range(1, 10)
# Fit a model for each lag length and compute the BIC
bics = []
for lag in lags:
    # Create the lagged variables
    df_lagged = pd.concat([df.shift(i) for i in range(lag+1)], axis=1)
    df_lagged.columns = ['y'] + ['L' + str(i) for i in range(1, lag+1)]    
    df_lagged = df_lagged.fillna(method = 'bfill')
    # Fit an ARIMA model with a constant term
    model = sm.tsa.ARIMA(df_lagged['y'], order=(lag, 0, 0), exog=exog_df, trend='c',freq='D')
    results = model.fit()

    # Compute the BIC
    bics.append(results.bic)

# Choose the lag length that minimizes the BIC
optimal_lag = lags[bics.index(min(bics))]
#####################################################################################################################################################################
#####################################################################################################################################################################
# Append the lags
def create_lagged_features(df, optimal_lag):
    for i in range(1, optimal_lag + 1):
        df[f'lag_{i}'] = df.C_TotalSales.shift(i)
    return df
#####################################################################################################################################################################
#####################################################################################################################################################################
df = create_lagged_features(merged_df.copy(), optimal_lag)
df.fillna(method='bfill',inplace=True)
df.isna().sum()
#####################################################################################################################################################################
#####################################################################################################################################################################
# Creating the train,val,test Datasets
train, val,test = df[:int(len(df) * 0.8)],df[int(len(df) * 0.8):int(len(df) * 0.9)],df[int(len(df) * 0.9):]
#####################################################################################################################################################################
#####################################################################################################################################################################
# Creating domain and Codomain datasets
X_train = train.drop(['C_TotalSales'], axis=1)
y_train = train['C_TotalSales']
X_val = val.drop(['C_TotalSales'], axis=1)
y_val = val['C_TotalSales']
X_test = test.drop(['C_TotalSales'], axis=1)
y_test = test['C_TotalSales']
#####################################################################################################################################################################
#####################################################################################################################################################################
# Pickel DataFrames for model
# pickle train dataframes to a file
with open("train_df.pkl", "wb") as f:
    pd.to_pickle(X_train, f)
with open("train_df.pkl", "wb") as f:
    pd.to_pickle(y_train, f)

# pickle validation dataframes to a file
with open("val_df.pkl", "wb") as f:
    pd.to_pickle(X_val, f)
with open("val_df.pkl", "wb") as f:
    pd.to_pickle(y_val, f)
    
# pickle test dataframes to a file
with open("test_df.pkl", "wb") as f:
    pd.to_pickle(X_test, f)
with open("test_df.pkl", "wb") as f:
    pd.to_pickle(y_test, f)
