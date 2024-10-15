# Klassificering blommor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Ladda in Iris-datasetet
# Klassiskt dataset för maskininlärning med information om olika irisblommor
# Finns tre olika klasser att dela in blommorna i: Setosa(0), Versicolor(1) och Virginica(2)
iris = load_iris()
X, y = iris.data, iris.target
# Varje sample har 4 features: sepal length, sepal width, petal length, petal width

# Dela upp datan i tränings- och testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisera indata
# viktigt för att få alla features i samma skala
X_train, X_test = X_train / np.max(X_train), X_test / np.max(X_test)

# Konvertera etiketter till one-hot encoding
# Nödvändigt för fler flerklass-klassificering
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Skapa en enkel sekventiell modell.
# Linjär stack av lager, detta vanliga klassificierings-problemet.
# Grundläggande typ av neural network, varje lager kopplat till nästa
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # Specificera input-shape = antal features för varje blomma
    tf.keras.layers.Dense(10, activation='relu',),      # 10 noder per lager, eftersom
    tf.keras.layers.Dense(10, activation='relu',),      # Fler lager, mer kapacitet att lära sig
    tf.keras.layers.Dense(3, activation='softmax',),    # Sista lagret har lika många noder som antalet target-klasser
])
# 1-3 "hidden layers" är rimligt i ett enkelt fall.
# Dense-lager är vanlig/typisk. I Dense-lager är varje nod kopplad till alla andra noder i det över lagret.
# Dense-lager ofta effektiva för strukturerad data.

# Här konfigurerar vi modellen inför träning.
# Optimizer: Adam - Adam är en adaptive learning rate optimization algorithm. Populär.
# Loss-funktion: categorical_crossentropy - Bra för multi-klass-klassificerings-problem.
# Metrik: accuracy - Hur många korrekta predictions vi gör.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Träna modellen
# Epochs = antalet gånger modellen går igenom hela datasetet under träning
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1) # Verbose låter oss se träningen.

# Utvärdera modellen på testdata
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

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
# Training Loss: Beräknas under varje epoch. Representerar hur bra model fitting.
# Validation Loss: Beräknas på separat validerings-data. Visar eventuell model overfitting.
# Om båda går nedåt under träning, lär sig modellen väl.
# Om training loss går ner, men validation loss börjar gå upp, kan det indikera overfitting 
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
sample_predictions = model.predict(X_test[:3])
print("\nSample predictions:")
for i, pred in enumerate(sample_predictions):
    print(f"Example {i+1}: {pred}")
    print(f"Predicted class: {np.argmax(pred)}")
    print(f"Actual class: {np.argmax(y_test[i])}")
    print()