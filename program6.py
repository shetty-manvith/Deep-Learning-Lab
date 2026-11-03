# Program 6 – LSTM Forecasting


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate time series data
data = np.sin(np.arange(1000) * 0.01)

# Prepare sequences
X = []
y = []

for i in range(50, len(data)):
    X.append(data[i-50:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

# Reshape input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(50, 1)),
    Dense(1)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train model
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32
)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
loss = model.evaluate(X_test, y_test)

print("Mean Squared Error:", loss)
print("Predictions shape:", predictions.shape)
print("LSTM Time Series Forecasting Completed!")
