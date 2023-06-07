import requests
import numpy as np
import re
import pandas as pd
import json
import statsmodels.api as sm
from pandas_market_calendars import get_calendar, MarketCalendar
from datetime import datetime, timedelta
import os
import pickle
import pgeocode
#####################################################################################################################################################################
#####################################################################################################################################################################
# Import Data
cwd = os.getcwd()
path = os.path.join(cwd, '')
# Customer Sales Query
file1 = "Query1.xlsx"
# Customer Postal Codes and tokens
file3 = "Customer_Token_PostalCodes.xlsx"
df_token = pd.read_excel(path + file3)
df_sales = pd.read_excel(path + file1)
# Sales DataFrame and customer tokens
df = pd.merge(df_sales, df_token, on='CustomerName', how='left')
df_token
df_sales
#####################################################################################################################################################################
#####################################################################################################################################################################
# Sales DataFrame and customer tokens
df = pd.merge(df_sales, df_token, on='CustomerName', how='left')
#####################################################################################################################################################################
#####################################################################################################################################################################
df = df.loc[df.token == '5D21088F-6603-4ECD-A4DA-237DE018C8FD']
#####################################################################################################################################################################
#####################################################################################################################################################################
# Sort by time and set index
df.set_index('SalesDate',inplace=True)
df.sort_index(inplace=True)
df.index
df = df.C_TotalSales.asfreq('D')
#####################################################################################################################################################################
#####################################################################################################################################################################
# Define start and end dates
startdate = df.SalesDate.min().strftime('%Y-%m-%d')
# '2022-06-01'
enddate = '2023-04-01'
# enddate = pd.Timestamp.now().date()
api_key = 'sLyRxkcsmqBlZUQtwoVxrIGLkTAbTosu'  
# postal_code = df.iloc[0].PostalCode  
postal_code = df.PostalCode.iloc[0]                                          
# Create a range of dates
date_range = pd.date_range(start=startdate, end=enddate)
# Create an empty dataframe with the date_range as index
ts_df = pd.DataFrame(index=date_range)
# Add datetime column to use 'dt' functino to retrieve seasonal data
ts_df['SalesDate'] = ts_df.index
# Preview the dataframe
print(ts_df.isna().sum())
#####################################################################################################################################################################
#####################################################################################################################################################################
# Add on columns for day of week and is or is not holiday
df_ts = ts_df.copy()
# Add a column for the day of the week
df_ts['day_of_week'] = ts_df['SalesDate'].dt.strftime('%A')
# Create a calendar object for the NYSE
nyse = get_calendar('NYSE')
# Get the holiday calendar for the NYSE
nyse_holidays = nyse.schedule(start_date=ts_df['SalesDate'].min(), end_date=ts_df['SalesDate'].max())
# Add a column for whether each date is a holiday
df_ts['is_holiday'] = ts_df['SalesDate'].isin(nyse_holidays.index)
#####################################################################################################################################################################
#####################################################################################################################################################################
# df = df.loc[df.token == '5D21088F-6603-4ECD-A4DA-237DE018C8FD']
# postal_code = df.PostalCode.iloc[0]
# Find 5 nearest stations using weather data textfile
def nearest_stations(postal_code,cwd):
    nomi = pgeocode.Nominatim('us')
    query = nomi.query_postal_code(postal_code)
    data = {"lat": query["latitude"],"lon": query["longitude"]}
    columns = ["ID", "LATITUDE", "LONGITUDE", "ELEVATION", "STATE", "NAME", "GSN_FLAG", "HCN_CRN_FLAG", "WMO_ID"]
    widths = [11, 9, 10, 7, 3, 31, 4, 4, 6]
    df_stations = pd.read_fwf(os.path.join(cwd, "ghcnd-stations.txt"), names=columns, header=None, widths=widths)
    df_stations["DISTANCE"] = ((df_stations["LATITUDE"] - data['lat']) ** 2 + (df_stations["LONGITUDE"] - data['lon']) ** 2) ** 0.5
    # Sort the DataFrame by distance and select the top 5 nearest stations
    nearest_stations = df_stations.nsmallest(5, "DISTANCE")
    nearest_stations = nearest_stations.ID.tolist()
    return nearest_stations
#####################################################################################################################################################################
#####################################################################################################################################################################
# Top 5 nearest stations
closest_stations = nearest_stations(postal_code,cwd)
closest_stations
#####################################################################################################################################################################
#####################################################################################################################################################################
# Weather data
def get_weather_data(station_id, api_key,startdate,enddate,type):
    endpoint = f'https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=GHCND:{station_id}&startdate={startdate}&enddate={enddate}&datatypeid={type}&limit=500'
    response = requests.get(endpoint, headers={'token': api_key})
    response.raise_for_status()
    data = response.json()
    return data
startdate,enddate
#####################################################################################################################################################################
#####################################################################################################################################################################
# Weather DataFrame
def is_complete_dataframe(df, start_date, end_date):
    date_range = pd.date_range(start=startdate, end=enddate, freq='D')
    return len(df) == len(date_range) and (df.index == date_range).all()
def weather_frame(name):
    weather_data = {}
    nearest_station_index = 0
    # name = 'PRCP'
    while not weather_data and nearest_station_index < len(closest_stations):
        station_id = closest_stations[nearest_station_index]
        data = get_weather_data(station_id, api_key, startdate, enddate, name)
        
        if 'results' in data:
            weather_data = data['results']
            df_weath = pd.DataFrame(weather_data)
            df_weath.set_index(pd.to_datetime(df_weath['date']), inplace=True)
            df_weath['value'] = df_weath['value']/10
            df_weath.rename(columns={'date': 'SalesDate', 'value': name}, inplace=True)
            df_weath = df_weath[['SalesDate',name]]
            if not is_complete_dataframe(df_weath, startdate, enddate):
                weather_data = {}
        
        nearest_station_index += 1

    if weather_data:
        return df_weath
    else:
        print("No full dataset found for the top 5 nearest stations.")

#####################################################################################################################################################################
#####################################################################################################################################################################
# Make temperature dataframe
dfPrcp = weather_frame('PRCP')
dfPrcp.drop('SalesDate',axis=1,inplace=True)
dfPrcp.isna().sum()
dfTmin = weather_frame('TMIN')
dfTmin.drop('SalesDate',axis=1,inplace=True)
dfTmin.isna().sum()
dfTmax = weather_frame('TMAX')
dfTmax.drop('SalesDate',axis=1,inplace=True)
dfTmax.isna().sum()
exog_df = pd.concat([df_ts,dfPrcp,dfTmin,dfTmax], axis=1)
exog_df.drop('SalesDate',axis=1,inplace=True)
exog_df.isna().sum()
full_df = pd.merge(df, exog_df, how='outer', left_index=True, right_index=True)
# Change categorical variables
dummies1 = pd.get_dummies(full_df.day_of_week)
dummies2 = pd.get_dummies(full_df.is_holiday)
# Concatenate the dummies to original dataframe
full_df = pd.concat([full_df, dummies1, dummies2], axis='columns')
# drop the values
full_df.drop(['day_of_week','is_holiday'], axis='columns',inplace=True)
#####################################################################################################################################################################
full_df['TAVG'] = (full_df.TMIN + full_df.TMAX)/2
full_df.columns
#####################################################################################################################################################################
#####################################################################################################################################################################
# Find Optimal lag
#Create Exogeneous vars
exog_df = full_df[['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',False,True,'PRCP','TMIN','TMAX','TAVG']]
#####################################################################################################################################################################
#####################################################################################################################################################################
# BIC Criterion to compute optimal lag 
# Restric to Daily Total Sales Column
# load dataset
df = full_df.C_TotalSales.copy()
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
df = create_lagged_features(full_df, optimal_lag)
df.isna().sum()
#####################################################################################################################################################################
#####################################################################################################################################################################
forecast_exogdf = full_df.loc['03-02-2023':]
forecast_exogdf = forecast_exogdf.drop(columns=['C_TotalSales','lag_1'])
train_df = full_df.loc[:'03-01-2023']
train_df.fillna(method='bfill',inplace=True)
train_df.isna().sum()
forecast_exogdf.isna().sum()

with open("ExogForecastData_03022023to04012023.pkl", "wb") as f:
    pd.to_pickle(forecast_exogdf, f)    
with open("TrainData_06012022to03012023.pkl", "wb") as f:
    pd.to_pickle(train_df, f)  
forecast = pd.read_pickle('ExogForecastData_03022023to04012023.pkl')
train = pd.read_pickle('TrainData_06012022to03012023.pkl')
forecast.isna().sum()
train.isna().sum()
    
    
    
    
    
    
    
    
    
