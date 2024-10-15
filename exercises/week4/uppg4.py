# Övning 4: Faltningsneuralt nätverk (CNN) med Keras

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Ladda och förbehandla data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)) / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Skapa CNN-modellen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Kompilera och träna modellen
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

# Utvärdera modellen
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc}")

# Visa några felklassificerade exempel
predictions = model.predict(X_test)
misclassified = np.where(np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1))[0]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified):
        idx = misclassified[i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {np.argmax(y_test[idx])}, Pred: {np.argmax(predictions[idx])}")
        ax.axis('off')
plt.tight_layout()
plt.show()

# Kommentarer:
# 1. MNIST-datasetet innehåller handskrivna siffror och är ett klassiskt dataset för bildklassificering.
# 2. Vi använder ett CNN för att lära oss hierarkiska representationer av bilderna.
# 3. Conv2D-lager extraherar lokala mönster, medan MaxPooling2D-lager reducerar dimensionaliteten.
# 4. Flatten-lagret omvandlar 2D-representationen till en 1D-vektor för de täta lagren.
# 5. Utgångslagret har 10 neuroner (en för varje siffra) med softmax-aktivering.
# 6. Vi visar några felklassificerade exempel för att förstå modellens misstag.