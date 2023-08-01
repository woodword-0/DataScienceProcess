import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Import data
# Import data
path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/SalesQuantity.xlsx"
df = pd.read_excel(path)
df.sort_values("Date",inplace=True)
#Prophet demands columns be designated as ds and y
df=df.rename(columns={'Date':'ds','Sales Quantity':'y'})
df.shape
df = df[df.y != 0]
# Create sequences of fixed length (e.g., 30 time steps)
sequence_length = 90

X, y = [], []
for i in range(len(df) - sequence_length):
    X.append(df['y'].iloc[i:i + sequence_length])
    y.append(df['y'].iloc[i:i + sequence_length].mean())

X = np.array(X)
y = np.array(y)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y.shape
actuals = df['y'].iloc[sequence_length:]
X.shape
actuals
plt.scatter(actuals,y)
plt.show()
y_forecasted = y
y_truth = actuals
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))
# df.index
# squared_error = (y - actuals) ** 2
# squared_error.index = df['ds'].iloc[sequence_length:]
# actuals.index
# squared_error.plot()
# plt.show()