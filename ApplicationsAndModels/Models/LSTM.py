import pandas as pd
import numpy as np
import warnings
# Reproducibility
import random
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Stats to get trend
from statsmodels.tsa.seasonal import seasonal_decompose

# scikit-learn 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Tensorflow (for LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
random.seed(42)
# Import Data
path = r"C:/Users/TurnerJosh/Desktop/SensitiveData/GoodSalesData1.xlsx"
df = pd.read_excel(path)

df = df[['Charge Amount','Invoice Date']].loc[df['Charge Amount'] != 0]
# Rename date series 'ds' and time series 'ts'
DF = df.rename(columns = {'Invoice Date': 'ds', 'Charge Amount': 'ts'})
# Outliers
DF = DF[(DF['ts'] < DF['ts'].mean() + 3*DF['ts'].std()) & (DF['ts'] > DF['ts'].mean() - 3*DF['ts'].std())]
# Prepare time series for LSTM 
DF.sort_values('ds',inplace=True)
sequence_length = 14

X, y = [], []
for i in range(len(DF) - sequence_length):
    X.append(DF['ts'].iloc[i:i + sequence_length])
    y.append(DF['ts'].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y.shape
X.shape
# Actuals
actuals = DF['ts'].values

# actuals = scaler.inverse_transform(actuals.reshape(-1, 1))
y_test_actuals = actuals[sequence_length:]
y_test_actuals = y_test_actuals[int(0.8*len(X)):]
y_test_actuals.shape,y.shape

#Train/Test
X_train,y_train,X_test,y_test = X[:int(0.8*len(X))],y[:int(0.8*len(X))],X[int(0.8*len(X)):],y[int(0.8*len(X)):]

# Create LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
plt.plot(y_train_pred)
plt.plot(y_test_pred)
plt.show()

# make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
plt.plot(y_train_pred)
plt.plot(y_test_pred)
plt.show()

# Evaluate Model
# train_rmse = np.sqrt(mean_squared_error(y_train_actuals, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actuals, y_test_pred))

print(f"Test RMSE: {test_rmse}")
