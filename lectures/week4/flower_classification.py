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

                # sepal length   | sepal width  | petal length    | petal width       | target/label
#blomma 1           5.1          | 3.5          | 1.4             | 0.2               | 0
#blomma 2           4.9          | 3.0          | 1.2             | 0.6               | 1
#blomma 3           4.5          | 4.0          | 1.4             | 0.9               | 0
...
#blomma 150         5.9          | 3.0          | 0.7             | 1.8               | 2



# Dela upp datan i tränings- och testset
train_test_split_data = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% testdata, 80% träningsdata

# X är datat, y är labels (korrekt output/prediction för det datan)
X_train = train_test_split_data[0] # Träningsdata
X_test = train_test_split_data[1]  # Testdata
y_train = train_test_split_data[2] # Tränings-labels
y_test = train_test_split_data[3]  # Test-labels
# Samma som:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisera indata
# viktigt för att få alla features i samma skala
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Konvertera etiketter till one-hot encoding
# Nödvändigt för fler flerklass-klassificering
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Skapa en enkel sekventiell modell.
# Linjär stack av lager, passar detta relativt simpla klassificierings-problemet.
# Grundläggande typ av neural network, varje lager kopplat till nästa
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(4,)),                  # Input-Layer: Specificera input-shape = antal features för varje blomma, d.v.s datats dimension = 4
    tf.keras.layers.Dense(10, activation='relu',),      # Hidden Layer 1: 10 noder per lager, bestäms efter experimentering
    tf.keras.layers.Dense(10, activation='relu',),      # Hidden Layer 2: Fler lager, mer kapacitet att lära sig
    tf.keras.layers.Dense(3, activation='softmax',),    # Output Layer: Sista lagret har lika många noder som antalet target-klasser, d.v.s modellen försöker välja mellan 3 kategorier för varje data-sample.
])
# 1-3 "hidden layers" är rimligt i ett enkelt fall.
# Dense-lager är vanlig/typisk. I Dense-lager är varje nod kopplad till alla andra noder i det över lagret.
# Dense-lager ofta effektiva för strukturerad data.
# Varje nod har en "aktiveringsfunktion" som appliceras på nodens output. Aktiveringsfunktionen är det som låter nätverket lära sig komplexa mönster i datat.
# Aktiveringsfunktionen hos en nod beräknar nodens output, baserat på dess olika inputs (från noderna i det föregående lagret) och deras vikt.
# Beroende på summan av viktade input-värden, plus en bias, avgör funktionen om noden ska aktiveras eller ej.
# Nodernas kopplingar är viktade, d.v.s de har ett tillhörande heltal som säger hur "stark" kopplingen mellan två noder är.
# activation='relu', ReLU (rectified linear unit), är den vanligaste aktiveringsfunktionen. Snabb, relativt simpel.
# activation='relu' Gör så att noderna har aktiveringsfunktionen ReLU, som ger en output x om x är positivt. Annars blir output 0.
# activation='softmax' använder vi för de tre noderna i sista lagret, Output Layer. Softmax är vanlig att använda i sista lagret i multi-class klassificerings-problem (t.ex. 3 klasser).
# activation='sigmoid' passar i binära klassficeringsproblem.

# Här konfigurerar vi modellen inför träning.
# Optimizer: Adam - Adam är en adaptive learning rate optimization algorithm. Populär.
# Valet av optimizer avgör hur vår modell ska ändra eller finjustera vårt neural networks egenskaper.
# Optimizerns mål är att under träningens gång förhöja modellens precision genom att ändra på till exempel vikter i nätverkets node layers.
# Loss-funktion: categorical_crossentropy - Bra för multi-klass-klassificerings-problem. Definierar och beräknar hur bra predictions vi gör. Vi vill minska 
# loss så myckety som möjligt.
# Cross-Entropy loss blir högre ju högre sannolikhet att modellen väljer fel kategori/klass.
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