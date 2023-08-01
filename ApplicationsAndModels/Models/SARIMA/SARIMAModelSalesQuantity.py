import itertools
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import calendar
import pickle
###############################################################################################################################################
###############################################################################################################################################
# Input data

path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/SalesQuantity.xlsx"
df = pd.read_excel(path)
df = df.sort_values('Date')
df.Date.nunique()
df = df.set_index('Date')
# df.describe()
###############################################################################################################################################
###############################################################################################################################################
# Rename Columns
DF = df.rename(columns = {'Sales Quantity': 'ts'})
DF = DF.resample('D').mean()
DF.describe()
# DF = DF.groupby(DF.index).agg({'ts':mean})
###############################################################################################################################################
###############################################################################################################################################
# DF.columns
# DF.isna().sum()
# df.shape
# DF.index
# DF.dropna(inplace=True)
# DF.index.inferred_freq
# DF_day = DF.resample('D')
# print(DF_day.shape)
###############################################################################################################################################

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
                                        #   exog = exog_df,
                                          order=param,
                                          seasonal_order=param_seasonal,
                                          enforce_stationarity=es,
                                          enforce_invertibility=False)
          results = mod.fit()
          if results.aic<aic:
            param1=param
            param2=param_seasonal
            aic=results.aic
            es1=es
          #print('ARIMA{}x{} enforce_stationarity={} - AIC:{}'.format(param, param_seasonal,es,results.aic))
        except:
          continue
# Save the model to disk using pickle
# with open('arima_model.pkl', 'wb') as f:
#     pickle.dump(results, f)
# print('Best model parameters: ARIMA{}x{} - AIC:{} enforce_stationarity={}'.format(param1, param2, aic,es1))
# Use optimal parameters to fit model
mod = sm.tsa.statespace.SARIMAX(ts,
                                order=param1,
                                seasonal_order=param2,
                                enforce_stationarity=es1,
                                enforce_invertibility=False)
results = mod.fit()
pred_uc = results.get_forecast(steps=30)

# pred_uc = results.get_forecast(steps=calendar.monthrange(datetime.now().year,datetime.now().month)[1]-datetime.now().day+1)
pred_ci = pred_uc.conf_int()
ax = ts.plot(label='observed', figsize=(12, 5))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
plt.legend()
plt.show()

predict=pred_uc.predicted_mean.to_frame()
predict.reset_index(inplace=True)
predict.rename(columns={'index': 'date',0: 'revenue_forcast'}, inplace=True)
display(predict)

results.plot_diagnostics(figsize=(20, 20))
plt.show()

y_forecasted = pred.predicted_mean
y_truth = ts['2022-06-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))
# Load the exogenous data
# with open('exog_df.pkl', 'rb') as f:
#     exog_df = pickle.load(f)
# exog_df

# # Generate a forecast
# with open('arima_model.pkl', 'rb') as f:
#     arima_model = pickle.load(f)
# arima_model

# forecast = arima_model.forecast(steps=14)
