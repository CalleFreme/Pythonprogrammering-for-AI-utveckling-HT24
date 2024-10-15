# Övning 10: Hyperparameterinställning med Scikit-learn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Ladda dataset
wine = load_wine()
X, y = wine.data, wine.target

# Dela upp data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiera parameterrutnät
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Skapa en RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Utför GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Skriv ut bästa parametrar och score
print("Bästa parametrar:", grid_search.best_params_)
print("Bästa cross-validation score:", grid_search.best_score_)

# Utvärdera den bästa modellen på testdata
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nKlassificeringsrapport för bästa modell:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Visualisera resultat
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 8))
plt.scatter(results['param_n_estimators'], results['mean_test_score'], c=results['param_max_depth'], cmap='viridis')
plt.colorbar(label='max_depth')
plt.xlabel('n_estimators')
plt.ylabel('Mean test score')
plt.title('GridSearchCV Results')
plt.show()

# Träna en modell med standardparametrar för jämförelse
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
print("\nKlassificeringsrapport för standardmodell:")
print(classification_report(y_test, y_pred_default, target_names=wine.target_names))

# Kommentarer:
# 1. Vi använder Wine-datasetet, som innehåller kemiska analyser av viner från olika kultivarer.
# 2. RandomForestClassifier används som vår modell. Den kombinerar flera beslutsträd för att göra prediktioner.
# 3. GridSearchCV används för att systematiskt gå igenom olika kombinationer av hyperparametrar.
# 4. Parameterrutnätet innehåller olika värden för n_estimators (antal träd), max_depth (maximalt träddjup), 
#    min_samples_split och min_samples_leaf (kontrollerar trädets komplexitet).
# 5. Cross-validation med 5 veck används för att utvärdera varje parameterkombination.
# 6. Resultaten visualiseras för att se hur olika parametrar påverkar modellens prestanda.
# 7. Vi jämför den optimerade modellen med en modell som använder standardparametrar för att se förbättringen.
# 8. Hyperparameterinställning är viktigt för att få bästa möjliga prestanda från en modell och undvika överfitting.