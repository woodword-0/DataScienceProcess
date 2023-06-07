import itertools
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import statsmodels.api as sm
from io import BytesIO
import plotly.graph_objs as go
import plotly.express as px
from flask import Flask, make_response, render_template

# Read data
df = pd.read_pickle('TrainData_06012022to03012023.pkl')

# Create exogenous variables
exog_df = df[['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday',
                False,True,'PRCP','TMIN','TMAX','TAVG','lag_1']]

# Rename columns
df = df.rename(columns={'C_TotalSales': 'ts'})

# Get optimal parameters for SARIMAX model
ts = df['ts']
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
pred_ci = pred_uc.conf_int()

# Create a Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_uc.predicted_mean.index,
                         y=pred_uc.predicted_mean.values,
                         name='Forecast'))
fig.add_trace(go.Scatter(x=pred_ci.index,
                         y=pred_ci.iloc[:, 0],
                         fill=None,
                         mode='lines',
                         line_color='gray',
                         showlegend=False))
fig.add_trace(go.Scatter(x=pred_ci.index,
                         y=pred_ci.iloc[:, 1],
                         fill='tonexty',
                         mode='lines',
                         line_color='gray',
                         showlegend=False))
fig.update_layout(title='Sales Forecast for March 2023',
                  xaxis_title='Date',
                  yaxis_title='Sales')

# Save the plot to a buffer
buffer = BytesIO()
buffer.write(fig.to_html().encode('utf-8'))

# fig.write_html(buffer)
buffer
# Create a Flask app
app = Flask(__name__)

# Define route to display forecast plot
@app.route('/')
def generate_forecast_plot():
    # Create a response object containing the plot image
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'text/html'
    html_str = response.data.decode()
    return render_template('plotly.html', plot=html_str)

if __name__ == '__main__':
    app.run(debug=True)

