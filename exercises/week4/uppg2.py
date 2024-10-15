# Övning 2: Binär klassificering med Keras

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Ladda datasetet
data = load_breast_cancer()
X, y = data.data, data.target

# Normalisera data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dela upp data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Skapa modellen
model = Sequential([
    Dense(16, activation='relu', input_shape=(30,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilera modellen
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Träna modellen
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Utvärdera modellen
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Testaccuracy: {accuracy}")

# Kommentarer:
# 1. Vi använder bröstcancerdatasetet, där målet är att klassificera tumörer som godartade eller elakartade.
# 2. StandardScaler används för att normalisera data, vilket är viktigt för neurala nätverk.
# 3. Vår modell har två dolda lager med ReLU-aktivering och ett utgångslager med sigmoid-aktivering för binär klassificering.
# 4. Binary crossentropy används som förlustfunktion, vilket är lämpligt för binära klassificeringsproblem.
# 5. Adam-optimeraren används för att uppdatera vikterna i nätverket under träning.
# 6. Accuracy används som utvärderingsmått för att mäta modellens prestanda.