# Övning 7: Dimensionalitetsreduktion med TensorFlow

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.manifold import TSNE

# Ladda data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Skapa autoencoder
input_dim = 784
encoding_dim = 32

input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Träna autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=0)

# Använd encoder för dimensionalitetsreduktion
encoded_imgs = encoder.predict(x_test)

# Visualisera med t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(encoded_imgs)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_test, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of encoded Fashion MNIST data')
plt.show()

# Kommentarer:
# 1. Vi använder Fashion MNIST-datasetet, som innehåller bilder av klädesplagg och accessoarer.
# 2. Autoencodern består av en encoder som komprimerar datan och en decoder som rekonstruerar den.
# 3. Encodern reducerar dimensionaliteten från 784 (28x28 pixlar) till 32.
# 4. Autoencodern tränas för att rekonstruera inmatningsbilderna, vilket tvingar den att lära sig en kompakt representation.
# 5. Vi använder den tränade encodern för att skapa lågdimensionella representationer av testdatan.
# 6. t-SNE används för att visualisera de 32-dimensionella representationerna i 2D.
# 7. Visualiseringen visar hur olika klädesplagg grupperas baserat på deras egenskaper i den lågdimensionella representationen.