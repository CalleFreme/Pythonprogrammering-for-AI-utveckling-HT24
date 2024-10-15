# Kom igång med Jupyter Notebooks och huspris-predictions

## Innehållsförteckning

1. [Introduktion](#introduktion)
2. [Installation](#installation)
3. [Starta Jupyter Notebook](#starta-jupyter-notebook)
4. [Skapa ett nytt projekt](#skapa-ett-nytt-projekt)
5. [Importera nödvändiga bibliotek](#importera-nödvändiga-bibliotek)
6. [Ladda in och förbereda data](#ladda-in-och-förbereda-data)
7. [Skapa en enkel modell](#skapa-en-enkel-modell)
8. [Träna modellen](#träna-modellen)
9. [Utvärdera modellen](#utvärdera-modellen)
10. [Gör prediktioner](#gör-prediktioner)
11. [Avslutning](#avslutning)

## Introduktion

Denna guide hjälper dig att komma igång med Jupyter Notebooks och skapa ett enkelt program för att förutsäga huspriser. Vi kommer att använda Python och några populära datavetenskap-bibliotek.

## Installation

För att komma igång behöver du installera följande:

1. Python (version 3.7 eller senare)
2. Jupyter Notebook
3. Pandas
4. NumPy
5. Scikit-learn

Du kan installera allt detta med pip:

```python
pip install jupyter pandas numpy scikit-learn
```

## Starta Jupyter Notebook

Öppna en terminal eller kommandotolken och skriv:

```python
jupyter notebook
```

Detta öppnar Jupyter Notebook i din webbläsare.

## Skapa ett nytt projekt

1. Klicka på "New" i övre högra hörnet
2. Välj "Python 3" under "Notebook"

## Importera nödvändiga bibliotek

I den första cellen, skriv följande kod:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

Kör cellen genom att trycka på "Run" eller använd kortkommandot Shift+Enter.

## Ladda in och förbereda data

I nästa cell, ladda in din data. För detta exempel antar vi att du har en CSV-fil med husdata:

```python
# Ladda in data
data = pd.read_csv('husdata.csv')

# Visa de första raderna
print(data.head())

# Välj features och target
X = data[['area', 'rum', 'ålder']]
y = data['pris']

# Dela upp data i tränings- och testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Skapa en enkel modell

Vi använder linjär regression för vår modell:

```python
# Skapa en linjär regressionsmodell
model = LinearRegression()
```

## Träna modellen

Nu tränar vi modellen på vårt träningsdata:

```python
# Träna modellen
model.fit(X_train, y_train)
```

## Utvärdera modellen

Låt oss utvärdera hur bra vår modell presterar:

```python
# Gör prediktioner på testdata
y_pred = model.predict(X_test)

# Beräkna mean squared error och R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")
```

## Gör prediktioner

Nu kan vi använda vår modell för att förutsäga huspriser:

```python
# Exempel: Förutsäg priset för ett hus med area 120 kvm, 3 rum och 5 år gammalt
nytt_hus = [[120, 3, 5]]
pris_prediktion = model.predict(nytt_hus)

print(f"Det förväntade priset för huset är: {pris_prediktion[0]:.2f} kr")
```

## Avslutning

Grattis! Du har nu skapat ett enkelt program för att förutsäga huspriser med hjälp av Jupyter Notebook och maskininlärning. Detta är bara början - du kan fortsätta att förbättra din modell genom att experimentera med olika algoritmer, lägga till fler features eller använda mer avancerade tekniker för databehandling och modellering.
