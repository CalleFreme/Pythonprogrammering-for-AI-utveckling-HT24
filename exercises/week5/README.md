# Vecka 5 - Supervised Learning

När du gör dessa uppgifter, experimentera gärna med olika hyperparametrar så som
Försök visualisera resultaten.
Försök argumentera för styrkor och svagheter med varje metod.
Använd Google för att hitta information om biblioteken och deras dataset och tillgängliga modeller/algoritmer.

## Övning 1: Linjär Regression med Scikit-learn

(Om du inte redan gjort denna uppgift)
Använd Scikit-learn för att implementera linjär regression på Kaliforniens husprisdataset.

1. Ladda California husprisdataset från Scikit-learn.
2. Dela upp data i tränings- och testset.
3. Skapa och träna en linjär regressionsmodell.
4. Förutsäg huspriser för testdatan och beräkna Mean Squared Error.
5. Visualisera de faktiska vs. förutsagda priserna med ett spridningsdiagram.

## Övning 2: Logistisk Regression för Binär Klassificering

Implementera logistisk regression för att klassificera iris-blommor.

1. Ladda iris-datasetet från Scikit-learn.
2. Välj endast två klasser för binär klassificering (t.ex. versicolor och virginica).
3. Dela upp data i tränings- och testset.
4. Skapa och träna en logistisk regressionsmodell.
5. Utvärdera modellen med accuracy, precision, recall och F1-score.

## Övning 3: k-Nearest Neighbors (k-NN) Klassificering

Använd k-NN för att klassificera handskrivna siffror.

1. Ladda MNIST-datasetet (eller en delmängd av det) från Scikit-learn.
2. Dela upp data i tränings- och testset.
3. Skapa en k-NN-klassificerare med olika värden på k (t.ex. 1, 3, 5, 7).
4. Träna och utvärdera modellerna.
5. Jämför resultaten för olika k-värden och visualisera några korrekt och felaktigt klassificerade exempel.

## Övning 4: Decision Tree för Klassificering

Implementera ett beslutsträd för att klassificera vintyper.

1. Ladda Wine-datasetet från Scikit-learn.
2. Dela upp data i tränings- och testset.
3. Skapa och träna ett beslutsträd.
4. Visualisera beslutsträdet.
5. Beräkna feature importance och visa de viktigaste attributen.

## Övning 5: Random Forest för Regression

Använd Random Forest för att förutsäga diabetes progression.

1. Ladda Diabetes-datasetet från Scikit-learn.
2. Dela upp data i tränings- och testset.
3. Skapa och träna en Random Forest Regressor.
4. Utvärdera modellen med Mean Absolute Error och R-squared.
5. Jämför resultaten med en enkel linjär regression.

## Övning 6: Support Vector Machine (SVM) för Klassificering

Implementera en SVM för att klassificera bröstcancer.

1. Ladda Breast Cancer Wisconsin-datasetet från Scikit-learn.
2. Dela upp data i tränings- och testset.
3. Skapa och träna en SVM-klassificerare med olika kernels (linjär, RBF).
4. Jämför resultaten för olika kernels.
5. Visualisera beslutsgränsen för den bästa modellen.

## Övning 7: Naive Bayes för Textklassificering

Använd Naive Bayes för att klassificera nyhetsartiklar.

1. Ladda 20 Newsgroups-datasetet från Scikit-learn (välj några få kategorier).
2. Förbehandla texten (tokenisering, borttagning av stopord, etc.).
3. Använd TF-IDF för feature extraction.
4. Implementera och träna en Multinomial Naive Bayes-klassificerare.
5. Utvärdera modellen och visa några korrekt och felaktigt klassificerade exempel.

## Övning 8: Gradient Boosting för Regression

Implementera Gradient Boosting för att förutsäga cykelhyror.

1. Ladda Bike Sharing-datasetet (finns på UCI Machine Learning Repository).
2. Förbehandla data (hantera kategoriska variabler, normalisera numeriska features).
3. Dela upp data i tränings- och testset.
4. Skapa och träna en Gradient Boosting Regressor.
5. Utvärdera modellen och visualisera feature importance.

## Övning 9: Multi-layer Perceptron (MLP) för Klassificering

Använd en enkel neural network (MLP) för att klassificera modetrender.

1. Ladda Fashion-MNIST-datasetet.
2. Normalisera bilddata.
3. Skapa en MLP-modell med Scikit-learn.
4. Träna modellen och utvärdera dess prestanda.
5. Experimentera med olika nätverksarkitekturer och aktiveringsfunktioner.

## Övning 10: Ensemble Learning

Implementera en Voting Classifier för att kombinera flera modeller.

1. Använd iris-datasetet igen, men denna gång med alla tre klasser.
2. Skapa tre olika klassificerare (t.ex. Logistic Regression, Decision Tree, och k-NN).
3. Kombinera dessa i en Voting Classifier.
4. Jämför prestandan för den kombinerade modellen med de individuella modellerna.
5. Experimentera med både hard och soft voting.
