# Övning 5: Textklassificering med TensorFlow och Keras

import numpy as np
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer

# Ladda data
vocab_size = 10000
max_length = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding av sekvenser
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Skapa modellen
model = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Kompilera och träna modellen
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)

# Utvärdera modellen
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc}")

# Funktion för att klassificera anpassade recensioner
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def classify_review(review):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([review])
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# Testa modellen på några anpassade recensioner
custom_reviews = [
    "This movie was fantastic! The acting was superb and the plot was engaging.",
    "I was very disappointed with this film. The story was boring and predictable."
]

for review in custom_reviews:
    print(f"Review: {review}")
    print(f"Sentiment: {classify_review(review)}\n")

# Kommentarer:
# 1. IMDB-datasetet innehåller filmrecensioner märkta som positiva eller negativa.
# 2. Vi använder ett inbäddningslager för att konvertera ord till täta vektorer.
# 3. LSTM-lager används för att fånga långsiktiga beroenden i texten.
# 4. Modellen avslutas med ett dense-lager med sigmoid-aktivering för binär klassificering.
# 5. pad_sequences används för att se till att alla inmatningssekvenser har samma längd.
# 6. Vi skapar en funktion för att klassificera anpassade recensioner, vilket visar modellens praktiska användning.