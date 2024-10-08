# Vecka 4 - Introduktion till AI i Python

## 1. AI och Machine Learning

### 1.1 Vad är AI och Machine Learning?

- Artificiell Intelligens (AI): Skapandet av intelligenta maskiner som kan utföra uppgifter som vanligtvis kräver mänsklig intelligens.
- Machine Learning (ML): En underkategori av AI där system lär sig från data utan att explicit programmeras.

### Viktiga termer och koncept

- Klassificering: En typ av övervakad inlärning där modellen lär sig att kategorisera indata i fördefinierade klasser (t.ex. spam/inte spam).
- Regression: En typ av övervakad inlärning där modellen förutsäger ett kontinuerligt numeriskt värde (t.ex. huspriser).
- Features: De egenskaper eller variabler i data som används för att göra förutsägelser.
- Etikett (Label): Det önskade utfallet eller målvariabeln som modellen försöker förutsäga.
- Träningsdata: Data som används för att lära modellen.
- Testdata: Data som används för att utvärdera modellens prestanda på ny, osedd data.
- Överfitting: När en modell lär sig träningsdata för väl och presterar dåligt på ny data.
- Underfitting: När en modell är för enkel för att fånga komplexiteten i data.

### 1.2 Typer av Machine Learning

1. Övervakad inlärning (supervised learning)
   - Klassificering: Förutsäga en kategori (t.ex. spam/inte spam)
   - Regression: Förutsäga ett kontinuerligt värde (t.ex. huspriser)
2. Oövervakad inlärning (unsupervised learning)
   - Klustring: Gruppera liknande datapunkter
   - Dimensionalitetsreduktion: Minska antalet variabler i data
3. Förstärkningsinlärning (reinforcement learning)
   - Lär sig genom interaktion med en miljö

### 1.3 Användningsområden

- Bildigenkänning och datorseende (image recognition and computer vision)
- Naturlig språkbehandling (NLP) (text recognition and speech recognition)
- Rekommendationssystem
- Autonoma fordon
- Medicinsk diagnos
- Finansiella prognoser

### 1.4 Tekniker i Machine Learning

- Neurala nätverk och djupinlärning (deep learning)
- Beslutsträd och slumpmässiga skogar (decision trees and random forests)
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Bayesianska metoder

### 1.5 Vanliga algoritmer

- Linjär regression
- Logistisk regression
- K-means klustring
- Principal Component Analysis (PCA)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)

## 2. Machine Learning i Python

### 2.1 Översikt över Python-bibliotek för ML

Python har blivit det dominerande språket för machine learning tack vare sitt rika ekosystem av bibliotek och verktyg.

### 2.2 TensorFlow

- Utvecklat av Google
- Används för storskalig maskininlärning och djupinlärning
- Stöder GPU-accelererad beräkning

#### Grundläggande användning av TensorFlow:

```python
import tensorflow as tf

# Skapa en enkel modell
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

# Kompilera modellen
model.compile(optimizer='adam', loss='mse')
```

### 2.3 Keras

- Högnivå-API för att bygga och träna djupa neurala nätverk
- Kan köras ovanpå TensorFlow, Theano eller CNTK

#### Exempel på att bygga en modell med Keras:

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### 2.4 Scikit-learn

- Omfattande bibliotek för klassisk maskininlärning
- Enkel och effektiv för dataanalys och databrytning
- Innehåller verktyg för förbehandling, modellval och utvärdering

#### Exempel på användning av Scikit-learn:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Antag att X är features och y är etiketter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Modellens noggrannhet: {accuracy}")
```

## 3. Förberedelser för praktiska övningar

### 3.1 Installation av nödvändiga bibliotek

Innan du börjar med övningarna, se till att du har följande bibliotek installerade:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### 3.2 Dataimport och -behandling

De flesta ML-projekt börjar med att importera och förbereda data:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ladda data
data = pd.read_csv('dataset.csv')

# Dela upp i features och målvariabel
X = data.drop('target', axis=1)
y = data['target']

# Dela upp i tränings- och testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisera data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3.3 Modellbyggnad och träning

Efter dataförberedelsen kommer modellbyggnad och träning:

```python
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Scikit-learn exempel
sk_model = LogisticRegression()
sk_model.fit(X_train_scaled, y_train)

# Keras exempel
keras_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
keras_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### 3.4 Modellutvärdering

Efter träning är det viktigt att utvärdera modellens prestanda:

```python
from sklearn.metrics import accuracy_score, classification_report

# Scikit-learn modell
sk_predictions = sk_model.predict(X_test_scaled)
print("Scikit-learn modellens noggrannhet:", accuracy_score(y_test, sk_predictions))
print(classification_report(y_test, sk_predictions))

# Keras modell
keras_predictions = (keras_model.predict(X_test_scaled) > 0.5).astype(int)
print("Keras modellens noggrannhet:", accuracy_score(y_test, keras_predictions))
print(classification_report(y_test, keras_predictions))
```

## 4. Praktiska övningar

Se de separata övningarna för hands-on erfarenhet med olika ML-tekniker och bibliotek. Dessa övningar kommer att ge dig praktisk erfarenhet av att använda TensorFlow, Keras och Scikit-learn för olika maskininlärningsuppgifter.

## 5. Etiska överväganden inom AI och ML

### 5.1 Vikten av etik i AI

- Rättvisa och bias i ML-modeller
- Transparens och förklarbarhet
- Dataintegritet och sekretess

### 5.2 Ansvarsfull AI-utveckling

- Regelverk och riktlinjer
- Etiska riktlinjer för AI-forskning och -utveckling

## 6. Framtida trender inom AI och ML

- Förstärkt och självövervakad inlärning
- AI för vetenskapliga upptäckter
- Kvantmaskininlärning
- AI i edge computing
