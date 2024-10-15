# Övning 8: Tidsserieprognos med Keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Skapa syntetisk tidsseriedata
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', end='2023-01-01', freq='M')
ts = pd.Series(np.sin(np.arange(len(dates)) * 0.1) + np.random.randn(len(dates)) * 0.2, index=dates)

# Förbehandla data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ts.values.reshape(-1, 1))

# Skapa sekvenser
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12
X, y = create_sequences(scaled_data, seq_length)

# Dela upp data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Bygg LSTM-modell
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Träna modellen
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Gör prediktioner
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invertera skalningen
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# Visualisera resultaten
plt.figure(figsize=(12, 6))
plt.plot(ts.index[seq_length:], scaler.inverse_transform(scaled_data[seq_length:]), label='Actual')
plt.plot(ts.index[seq_length:train_size], train_predict, label='Train Predict')
plt.plot(ts.index[train_size+seq_length:], test_predict, label='Test Predict')
plt.legend()
plt.title('Time Series Forecasting')
plt.show()

# Kommentarer:
# 1. Vi skapar en syntetisk tidsserie med en sinusvåg och lite brus för att simulera säsongsvariationer och oregelbundenheter.
# 2. MinMaxScaler används för att normalisera datan mellan 0 och 1, vilket är viktigt för LSTM-nätverk.
# 3. Vi skapar sekvenser av data för att träna LSTM-modellen, där varje sekvens är 12 månader och målet är nästa månads värde.
# 4. LSTM-lagret används för att fånga tidsberoenden i datan.
# 5. Modellen tränas på träningsdatan och utvärderas på testdatan.
# 6. Vi visualiserar de faktiska värdena tillsammans med prediktionerna för både tränings- och testdata.
# 7. Denna metod kan användas för att förutsäga framtida värden i tidsserien baserat på historiska mönster.