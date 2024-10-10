# Klassificering blommor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ladda in Iris-datasetet
# Klassiskt dataset för maskininlärning med information om olika irisblommor
iris = tf.keras.datasets.iris
(x_train, y_train), (x_test, y_test) = iris.load_data()

# Normalisera indata
# viktigt för att få alla features i samma skala
x_train, x_test = x_train / np.max(x_train), x_test / np.max(x_test)

# Konvertera etiketter till one-hot encoding
# Nödvändigt för fler flerklass-klassificering
y_train = tf.keras.utils.to_categorically(y_train)
y_test = tf.keras.utils.to_categorically(y_test)

# Skapa en enkel sekventiell modell
# Grundläggande typ av neural network, varje lager kopplat till nästa
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activatoin='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activatoin='relu',),
    tf.keras.layers.Dense(10, activatoin='softmax',),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Träna modellen
# Epochs = antalet gånger modellen går igenom hela datasetet under träning
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, verbose=0)

# Utvärdera modellen på testdata
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f'Test Accuracy: {test_acc}')

# Visualisera träningshistorik
plt.figure(figsize=(12, 4))

# Plot träningsnoggrannhet och valideringsnoggrannhet
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot träningsförlust och valideringsförlust
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Gör prediktioner på några exempel
sample_predictions = model.predict(x_test[:3])
print("\nSample predictions:")
for i, pred in enumerate(sample_predictions):
    print(f"Example {i+1}: {pred}")
    print(f"Predicted class: {np.argmax(pred)}")
    print(f"Actual class: {np.argmax(y_test[i])}")
    print()