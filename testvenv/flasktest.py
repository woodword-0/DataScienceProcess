import itertools
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
plt.style.use('fivethirtyeight')
from flask import Flask, make_response
import calendar
import pickle
###############################################################################################################################################
###############################################################################################################################################
df = pd.read_pickle('TrainData_06012022to03012023.pkl')
df.isna().sum()
#Create Exogeneous vars
exog_df = df[['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',
                False,True,'PRCP','TMIN','TMAX','TAVG','lag_1']]
###############################################################################################################################################
###############################################################################################################################################
# Rename Columns
DF = df.rename(columns = {'C_TotalSales': 'ts'})
###############################################################################################################################################
###############################################################################################################################################
DF.columns
###############################################################################################################################################

df.index


ts = DF['ts']

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
aic = float('inf')
for es in [True,False]:
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

# Make predictions for March 2023
pred_uc = results.get_forecast(steps=31)
pred_uc
pred_ci = pred_uc.conf_int()
ax = ts.plot(label='observed', figsize=(12, 5))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
plt.legend()
plt.show()

# Create a Flask app
app = Flask(__name__)

# Define route to display forecast plot
@app.route('/')
def generate_forecast_plot():
    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Create a response object containing the plot image
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == '__main__':
    app.run(debug=True)
