import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download stock market data
data = yf.download('AAPL', start="2010-01-01", end="2022-12-31")[['Close']]

# Preprocess data
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
scaler = MinMaxScaler()
train_scaled, test_scaled = scaler.fit_transform(train), scaler.transform(test)

# Create dataset function
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Prepare datasets
time_step = 60
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)
X_train, X_test = X_train.reshape(-1, time_step, 1), X_test.reshape(-1, time_step, 1)

# Build and train LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Make and inverse transform predictions
train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

# Plot predictions
def plot_predictions(data, train_predict, test_predict, time_step):
    predictions = np.empty_like(data)
    predictions[:, :] = np.nan
    predictions[time_step:len(train_predict) + time_step, :] = train_predict
    test_start_idx = len(train_predict) + (time_step * 2)
    predictions[test_start_idx:test_start_idx + len(test_predict), :] = test_predict
    plt.plot(data, label='Original Data')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()

plot_predictions(data.values, train_predict, test_predict, time_step)
