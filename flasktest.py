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
import os

cwd = os.getcwd()
print("Current working directory:", cwd)
# import os

# os.chdir('SARIMAModel')

###############################################################################################################################################
###############################################################################################################################################
df = pd.read_pickle('TrainData_06012022to04012023.pkl')
df.isna().sum()
df.fillna(method='bfill',inplace=True)
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

# Make predictions fon 31 days
pred_uc = results.get_forecast(steps=31)
pred_uc
# Get predicted sales data as a pandas DataFrame
sales_pred = pred_uc.predicted_mean.to_frame(name='Predicted Sales')

# Save the predicted sales data to an Excel file
sales_pred.to_excel('predicted_sales.xlsx')
# Save the predicted sales data to an Excel file in a specific directory
# sales_pred.to_excel('/path/to/directory/predicted_sales.xlsx')

pred_ci = pred_uc.conf_int()
ax = ts.plot(label='observed', figsize=(12, 5))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
plt.legend()
plt.show()
# Save the plot to a buffer
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)

# Create a Flask app
app = Flask(__name__)

# Define route to display forecast plot
@app.route('/')
def generate_forecast_plot():
    # Create a response object containing the plot image
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == '__main__':
    app.run(debug=True)



# import streamlit as st
# import numpy as np
# import time
# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))
# pred_uc
# for i in range(100):
#     # Update progress bar.
#     progress_bar.progress(i + 1)

#     new_rows = np.random.randn(10, 2)

#     # Update status text.
#     status_text.text(
#         'The latest random number is: %s' % new_rows[-1, 1])

#     # Append data to the chart.
#     chart.add_rows(new_rows)

#     # Pretend we're doing some computation that takes time.
#     time.sleep(0.1)

# status_text.text('Done!')
# st.balloons()