import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools

# Set Streamlit app title
st.title("SARIMA Model Prediction")

# Load the data
df = pd.read_pickle('TrainData_06012022to04012023.pkl')
df.fillna(method='bfill', inplace=True)

# Create exogenous variables
exog_df = df[['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',
              False,True,'PRCP','TMIN','TMAX','TAVG','lag_1']]

# Rename Columns
df = df.rename(columns={'C_TotalSales': 'ts'})

# Fit SARIMA model
ts = df['ts']
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
aic = float('inf')

for es in [True, False]:
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                exog=exog_df,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=es,
                                                enforce_invertibility=False)
                results = mod.fit()
                if results.aic < aic:
                    param1 = param
                    param2 = param_seasonal
                    aic = results.aic
                    es1 = es
            except:
                continue

# Use optimal parameters to fit model
mod = sm.tsa.statespace.SARIMAX(ts,
                                order=param1,
                                seasonal_order=param2,
                                enforce_stationarity=es1,
                                enforce_invertibility=False)
results = mod.fit()

# Make predictions for 31 days
pred_uc = results.get_forecast(steps=31)
sales_pred = pred_uc.predicted_mean.to_frame(name='Predicted Sales')
pred_ci = pred_uc.conf_int()

# Plot the forecasted sales
fig, ax = plt.subplots(figsize=(12, 5))
ts.plot(label='Observed')
sales_pred.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
plt.legend()
plt.show()
st.pyplot(fig)  # Display the plot in Streamlit

# Display the predicted sales data as a table
st.subheader('Predicted Sales Data')
st.write(sales_pred)

# Save the predicted sales data to an Excel file
sales_pred.to_excel('predicted_sales.xlsx')