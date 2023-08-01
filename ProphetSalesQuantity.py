import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import floor
from prophet import Prophet
import os 
np.random.seed(9001)

# Import data
path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/SalesQuantity.xlsx"
df = pd.read_excel(path)

df.sort_values("Date",inplace=True)
#Prophet demands columns be designated as ds and y
df=df.rename(columns={'Date':'ds','Sales Quantity':'y'})
# df = df[df.y !=0]

# Train and Test Set
trainInd=floor(len(df)*.8)
valInd=floor(len(df)*.9)
train=df.loc[:trainInd,:]
val=df['ds'][trainInd:valInd]
val=val.to_frame()
test=df['ds'][valInd:]
test=test.to_frame()

# Model and Training
model = Prophet()
model.fit(train)
# Forecasting
forecast=model.predict(val)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# model.plot(forecast)
# plt.show()

val_y=df['y'][trainInd:valInd]
y_hat=forecast.yhat
val_y=val_y.to_numpy()
y_hat=y_hat.to_numpy()

# Model Fit
rss=((val_y-y_hat)**2).sum()
mse=np.mean((val_y-y_hat)**2)
print("Final mse value is:",mse)
print("Final rmse value is:",np.sqrt(np.mean((val_y-y_hat)**2)))


