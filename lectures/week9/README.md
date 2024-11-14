# Week 9 - Model Evaluation and Comparison

Utvärdering (evaluation) och jämförelse av algoritmer och modeller är kritiskt för att
förstå modellers prestanda, deras svagheter och styrkor, och för att förbättra resultat.

Beroende på typ av machine learning, tillämpar vi olika men principiellt liknande metoder för utvärdering och prestandamätning.

## Utvärdering i Supervised Learning

Att vi använder labeled data i supervised learning innebär att våra evaluation methods fokuserar på
att jämföra predicted labels med korrekta, faktiska labels.

### Classification Evaluation Metrics

Inom klassificering (binära fall eller flera klasser), används ofta:

- **Accuracy**: Andel korrekt klassificerade inputs
- **Precision, Recall, och F1-Score**: Mätvärden fokuserade kring "true/false positives/negatives", där *precision* syftar på "false positives", *recall* syftar på "false negatives", och *F1-Score* är en balans av de två.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Bra för binära klassificerings-algoritmer, där vi mäter modellens förmåga att separera klasser korrekt.

**Exempel:**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```

### Regression Evaluation Metrics

Inom regressionsproblem (prediction av kontinuerliga, numeriska värden), används ofta:

- **Mean Squared Error (MSE)**: Genomsnittet av kvadreterna av errors/fel.
- **Mean Absolute Error (MAE)**: Genomsnittet av absoluta värden av errors.
- **R-squared (R^2)**: Indikerar hur väl modellen förklarar variansen i vår target variable.
- **Root Mean Squared Error (RMSE)**: Roten (square root) av MSE. Bra för att tolka errors i den ursprungliga skalan.

**Exempel:**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared:", r2_score(y_test, y_pred))
```

### Cross-Validation (CV)

Ger mer robust modellutvärdering genom upprepad tränings och testning av modellen över olika uppdelningar (splits) av datat.
Vanliga metoder är:

- **k-Fold Cross-Validation:** Splittar data in i *k* grupper ("folds"), tränar på *k - 1* grupper och testar på den sista resterande gruppen, iterativt.
- **Stratified k-Fold:**: Bra för obalanserade dataset, säkerställer att varje fold har liknande fördelning av klasser (class distribution).
- **Leave-One-Out Cross-Validation (LOOCV):**: Använder alla datapunkter utom en för träning, och den ensamma datapunkter till testning. Repeteras för varje datapunkt.

**Exempel:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 5-Fold Cross-Validation
model = RandomForestClassifier()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Average Accuracy:", cv_scores.mean())
```

## Utvärdering i Unsupervised Learning

När vi saknar labeled data, och vi använder klustring, används ofta:

- **Silhouette Score**: Mäter hur pass lika datapunkter är inom ett kluster jämfört med andra kluster.
- **Davies-Bouldin Index**: Utvärderar klustrens compactness och separation.
- **Intertia (Sum of Squared Distances)**: Lägre inertia-värdenindikerar bättre cluster compactness.

**Exempel (klustring):**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Generate synthetic data for clustering
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

# Fit a KMeans model
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_

# Calculate metrics
print("Silhouette Score:", silhouette_score(X, labels))
print("Davies-Bouldin Score:", davies_bouldin_score(X, labels))
```

## Tekniker för modell-jämförelse

### Grid Search och Hyperparamter Tuning

För att komma fram till den bästa modellen eller konfigurationen för en modell, används ofta *grid search* eller *randomized search* för att systematiskt utforska optimala värden på hyperparametrar.

**Exempel:**

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)
```

### Ensemble och Stacking

Kombinering av modeller (ensemble methods) kan förbättra prediction quality. Vi kan kombinera och blanda modeller med tekniker så som *bagging*, *boosting*, eller *stacking*.

**Exempel (Voting Classifier):**

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Define individual classifiers
model1 = RandomForestClassifier()
model2 = GradientBoostingClassifier()
model3 = SVC(probability=True)

# Define ensemble model
voting_model = VotingClassifier(estimators=[
    ('rf', model1), ('gb', model2), ('svc', model3)], voting='soft')

# Train and evaluate
voting_model.fit(X_train, y_train)
print("Voting Model Accuracy:", voting_model.score(X_test, y_test))
```

### Tolkningsbarhet (Interpretability) och förklarbarhet (Explainability) hos modeller

För att förstå komplexa modeller, inte minst inom deep learning, används ofta interpretability-verktyg så som:

- **SHAP (SHapley Additive exPlanations)**: Mäter hur varje feature bidrar till modellens förståelse
- **LIME (Local Interpretable Model-agnostic Explanations)**: Generarar tolkningar/förklaringar lokalt runt varje prediction.

**Exempel (SHAP):**

```python
import shap

# Train a model
model = RandomForestClassifier().fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)
```

## Utvärdering i Reinforcement Learning

I RL baserar vi utvärderingar på vår implementations belöningssystem; ofta används:

- **Cumulative Reward:** Mäter den generella prestandan baserat på rewards.
- **Average Reward per Episode:** Mäter effektivitet och stabiliteten av inlärningen.
- **Steps per Episode:** Avgör hur snabbt en agent når sitt mål eller ett stabilt state.
