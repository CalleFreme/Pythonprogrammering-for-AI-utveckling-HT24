# Övning 1: Linjär regression med Scikit-learn

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_log_error

# Ladda datasetet
housing = fetch_california_housing()
X, y = housing.data, housing.target # X = features, y = target. Target är huspriser.

# Dela upp data i tränings- och testset. Vi använder 20% av data i testsetet.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Skapa och träna modellen
model = LinearRegression()
model.fit(X_train, y_train) # Tränar på träningsdata

# Gör förutsägelser och beräkna MSE
y_pred = model.predict(X_test) # Använd testdata i prediktion
mse = root_mean_squared_log_error(y_test, y_pred) # Jämför hur nära vår prediction kom de verkliga priserna i testdatat.

print(f"Genomsnittligt kvadratfel (MSE): {mse}")

# Kommentarer:
# 1. Vi använder California Housing-datasetet som innehåller information om bostäder i Kalifornien.
# 2. Linjär regression är en enkel men kraftfull metod för att modellera linjära samband mellan variabler.
# 3. Vi delar upp data i tränings- och testset för att utvärdera modellens prestanda på osedd data.
# 4. LinearRegression-klassen från Scikit-learn implementerar minsta kvadratmetoden för att hitta den bästa passningen.
# 5. (Root) Mean Squared (Log) Error (MSE) används som utvärderingsmått. Lägre MSE indikerar bättre modellprestanda.