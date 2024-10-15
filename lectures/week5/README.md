# Föreläsning vecka 5 - Supervised Learning

## Introduktion

* Definition: En ML-paradigm där modellen lär sig från labeled data
* Mål: Lär sig att koppla input-variabler till output-värde eller kategori.

## Viktiga koncept

* Training Data: Märkt (labeled) dataset där varje input är parad med korrekt output.
* Features: Input-variabler som vi använder för att göra predictions.
* Labels: Output-värde eller kategori som modellen ska förutsäga.
* Model: Funktion som lär sig koppla inputs till outputs.
* Loss Function: Mäter avståndet mellan predicted output och faktisk output.
* Optimization: Den process som minimerar loss funktion.

## Typer av problem inom Supervised Learning

1. **Classification**: Förutsäg diskreta kategorier/labels/klasser
    * Binär klassificering (t.ex. spam/inte spam)
    * Multi-klass klassificering (t.ex. känn igen handskriva siffror)
2. **Regression**: Försäg kontinuerliga värden (t.ex. huspriser, temperatur)
    * Linear Regression
    * Polynomial Regression

## Vanliga algoritmer

1. **Linear Regression**
Används för att förutsäga kontinuerliga värden.

2. **Logistic Regression** (classification)
Används för att förutsäga en binär utkomst, kategori. t.ex spam eller inte spam.

3. **Decision Trees** (classification+regression)
Trä-lik struktur av noder, som modellerar olika beslut och deras möjliga konsekvenser.
Kan användas för komplexa relationer mellan input features och output-värden.

4. **Random Forests** (classification+regression)
En grupp av decision trees som jobbar tillsammans.

5. **k-Nearest Neighbors (k-NN)** (classification+regression)
Hittar k antal närmsta grannarna närmast en given input,
förutsäger en kategori eller värde baserat majoritetens klass eller deras genomsnittliga värden.

6. Super Vector machines (SVM)
Skapar ett slags hyperplane att för att dela upp den n-dimensionella datan in i klasser.

7. Neural Networks
Deep learning kan kombineras med supervised learning.

8. **Gradient Boosting** (classification+regression)
Kombinerar "weak learners" så som decision trees, för att skapa en strong learner. 
Bygger nya modeller baserat på tidigare modeller, rättar fel från föregående.

9. **Naive Bayes Algorithm** (classification+regression)
Använder Bayes theorem, utgår "naivt" från att features är oberoende av varandra givet en label.
Funkar bra för t.ex. text-klassificering, spam, dokomunt-kategorisering.

## Fördelar med Supervised Learning

* Tydlig tolkning av modell-ouput
* Lätt att mäta modellens prestanda
* Stor bredd av algoritmer

## Utmaningar och begränsningar

* Kräver stora mängder märkt data
* Kan bli tunga beräkningar
* Risk för overfitting
* Svårt att hantera komlexa, icke-linjära relationer (för vissa algoritmer)

## Evaluation Metrics (Utvärdering av modellen)

* Classification: Accuracy, Precision, Recall, F1-score, ROC curve
* Regression: Mean squared error, R-squared, Mean Absolute Error (MAE)

## Best Practices

1. Förberedning av data: cleaning, normalizatoin, feature scaling
2. Feature Selection, feature Engineering
3. Cross-validation
4. Hyperparamter tuning
5. Enemble methods

## Supervised Learning i verkligeheten

* Bild- och röst-igenkänning
* Natural Language Processing
* Medicinsk diagnos
* Finansiella prognoser
* Rekommendations-system

## Supervised Learning I Python
