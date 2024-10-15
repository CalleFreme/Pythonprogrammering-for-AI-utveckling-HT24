# Övning 9: Överföringslärande med TensorFlow och Keras

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# Ladda förtränad VGG16-modell
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Frys de förtränade lagren
for layer in base_model.layers:
    layer.trainable = False

# Skapa ny modell på toppen av den förtränade
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')  # 5 klasser för blomdata
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dataugmentering och inläsning
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'path/to/flower/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path/to/flower/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Träna modellen
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=20
)

# Plotta träningshistorik
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Finjustera modellen
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Träna modellen igen med finjustering
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=10
)

# Kommentarer:
# 1. Vi använder en förtränad VGG16-modell som bas för vårt överföringslärande.
# 2. De förtränade lagren fryses initialt för att bevara deras inlärda funktioner.
# 3. Vi lägger till nya lager på toppen av VGG16 för att anpassa den till vår specifika uppgift (blomklassificering).
# 4. ImageDataGenerator används för dataugmentering, vilket hjälper till att förhindra överfitting.
# 5. Modellen tränas först med frysta baslager och sedan finjusteras genom att tina upp de sista lagren.
# 6. Överföringslärande utnyttjar kunskap från en relaterad uppgift (ImageNet) för att förbättra prestandan på vår specifika uppgift.
# 7. Denna metod är särskilt användbar när man har en liten datamängd för den specifika uppgiften.