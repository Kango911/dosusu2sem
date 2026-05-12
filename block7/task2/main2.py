# 2.3. Алгоритмы машинного обучения для классификации состояний

# ... (текстовое описание методов: 1D-свёртки, LSTM, ансамбли)

# Пример тренировочного кода для LSTM (умозрительный)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Допустим, что у нас есть временные ряды из 1000 окон по 50 отсчётов (3 датчика: давление P, расход Q, угол отвала α)
# Каждому окну соответствует метка: "норма" (0) или "неисправность" (1)

X = np.random.randn(1000, 50, 3)  # случайные демо-данные
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Масштабирование для каждого признака (в реальности лучше масштабировать по окну или по всей выборке)
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, 3)
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(-1, 50, 3)
X_test_reshaped = X_test.reshape(-1, 3)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(-1, 50, 3)

model = Sequential([
    LSTM(64, input_shape=(50, 3), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {acc:.3f}")