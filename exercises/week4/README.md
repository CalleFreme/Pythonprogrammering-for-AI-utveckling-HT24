# Vecka 4 - Maskininlärningsövningar med TensorFlow, Keras och Scikit-learn

Dessa övningar är till för att hjälpa dig lära dig och öva på att använda TensorFlow, Keras och Scikit-learn för olika maskininlärningsuppgifter. Vi börjar med enklare övningar och ökar gradvis komplexiteten.

## Förberedelser

Läs README-filen under lectures/week4 innan du börjar med dessa uppgifter.
Denna [Geeksforgeeks-länk](https://www.geeksforgeeks.org/python-ai/) är en av många bra resurser.

Innan du börjar med övningarna, se till att du har följande bibliotek installerade:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Övning 1: Linjär regression med Scikit-learn

**Data**: Använd Boston Housing-datasetet från Scikit-learn.

**Uppgift**: Skapa en enkel linjär regressionsmodell för att förutsäga huspriser.

**Steg**:

1. Ladda Boston Housing-datasetet med `sklearn.datasets.load_boston()`.
2. Dela upp data i tränings- och testset.
3. Skapa en `LinearRegression`-modell och anpassa den till träningsdata.
4. Gör förutsägelser på testsettet och beräkna genomsnittligt kvadratfel (MSE).

## Övning 2: Binär klassificering med Keras

**Data**: Använd Breast Cancer Wisconsin-datasetet från Scikit-learn.

**Uppgift**: Bygg ett neuralt nätverk för binär klassificering för att förutsäga om en tumör är elakartad eller godartad.

**Steg**:

1. Ladda Breast Cancer-datasetet med `sklearn.datasets.load_breast_cancer()`.
2. Normalisera egenskapsdata med `sklearn.preprocessing.StandardScaler`.
3. Skapa en sekventiell modell i Keras med två täta lager och en binär output.
4. Kompilera modellen med binär korsentropiförlust och träna den på data.
5. Utvärdera modellens noggrannhet på ett testset.

## Övning 3: Flerklass-klassificering med TensorFlow

**Data**: Använd Iris-datasetet från Scikit-learn.

**Uppgift**: Skapa en TensorFlow-modell för flerklass-klassificering för att förutsäga arten av irisar.

**Steg**:

1. Ladda Iris-datasetet med `sklearn.datasets.load_iris()`.
2. One-hot-koda målvariabeln.
3. Bygg en TensorFlow-modell med Keras API med lämpliga lager för flerklass-klassificering.
4. Träna modellen och plotta träningshistoriken (noggrannhet och förlust).
5. Gör förutsägelser på ett testset och skapa en förvirringsmatris.

## Övning 4: Faltningsneuralt nätverk (CNN) med Keras

**Data**: Använd MNIST-datasetet, som är inbyggt i Keras.

**Uppgift**: Bygg en CNN för att klassificera handskrivna siffror.

**Steg**:

1. Ladda MNIST-datasetet med `keras.datasets.mnist.load_data()`.
2. Förbehandla data (normalisera pixelvärden och omforma för CNN-input).
3. Skapa en CNN-modell med faltnings-, poolning- och täta lager.
4. Kompilera och träna modellen.
5. Utvärdera modellens prestanda och visa några felklassificerade exempel.

## Övning 5: Textklassificering med TensorFlow och Keras

**Data**: Använd IMDB Movie Review-datasetet, tillgängligt i Keras.

**Uppgift**: Skapa en modell för att klassificera filmrecensioner som positiva eller negativa.

**Steg**:

1. Ladda IMDB-datasetet med `keras.datasets.imdb.load_data()`.
2. Förbehandla data (padding av sekvenser till en fast längd).
3. Bygg en modell med ett inbäddningslager, LSTM-lager och ett tätt lager.
4. Träna modellen och utvärdera dess prestanda.
5. Använd modellen för att klassificera några anpassade filmrecensioner.

## Övning 6: Klustring med Scikit-learn

**Data**: Generera syntetisk data med `sklearn.datasets.make_blobs()`.

**Uppgift**: Utför K-means-klustring på den genererade datan.

**Steg**:

1. Generera en syntetisk dataset med 3 kluster med `make_blobs()`.
2. Implementera K-means-klustring med `sklearn.cluster.KMeans`.
3. Visualisera resultaten med ett spridningsdiagram med olika färger för varje kluster.
4. Experimentera med olika antal kluster och diskutera resultaten.

## Övning 7: Dimensionalitetsreduktion med TensorFlow

**Data**: Använd Fashion MNIST-datasetet, tillgängligt i Keras.

**Uppgift**: Implementera en autoencoder för dimensionalitetsreduktion och visualisering.

**Steg**:

1. Ladda Fashion MNIST-datasetet med `keras.datasets.fashion_mnist.load_data()`.
2. Bygg en autoencoder-modell med en encoder (reducerar dimensioner) och en decoder.
3. Träna autoencodern för att rekonstruera inputbilderna.
4. Använd encoder-delen för att reducera dimensionaliteten i datasetet.
5. Visualisera de reducerade representationerna med t-SNE eller PCA.

## Övning 8: Tidsserieprognos med Keras

**Data**: Använd Air Passengers-datasetet (tillgängligt online eller skapa en syntetisk tidsserie).

**Uppgift**: Bygg en LSTM-modell för att förutsäga framtida värden i tidsserien.

**Steg**:

1. Ladda och förbehandla tidsseriedata.
2. Skapa sekvenser av input-output-par för träning.
3. Bygg en LSTM-modell för tidsserieprediktion.
4. Träna modellen och gör framtida förutsägelser.
5. Visualisera de faktiska vs. förutsagda värdena.

## Övning 9: Överföringslärande med TensorFlow och Keras

**Data**: Använd en delmängd av Flowers-datasetet eller något litet bilddataset.

**Uppgift**: Implementera överföringslärande med en förtränad modell (t.ex. VGG16 eller ResNet50) för bildklassificering.

**Steg**:

1. Ladda en förtränad modell (t.ex. VGG16) utan topplagerna.
2. Lägg till nya lager för din specifika klassificeringsuppgift.
3. Frys de förtränade lagren och träna endast de nya lagren.
4. Finjustera modellen genom att tina upp några av de förtränade lagren.
5. Utvärdera modellens prestanda och jämför den med träning från grunden.

## Övning 10: Hyperparameterinställning med Scikit-learn

**Data**: Använd Wine-datasetet från Scikit-learn.

**Uppgift**: Utför hyperparameterinställning för en Random Forest Classifier.

**Steg**:

1. Ladda Wine-datasetet med `sklearn.datasets.load_wine()`.
2. Dela upp data i tränings- och testset.
3. Definiera ett parameterrutnät för Random Forest Classifier.
4. Använd `GridSearchCV` eller `RandomizedSearchCV` för att hitta de bästa hyperparametrarna.
5. Träna en slutlig modell med de bästa hyperparametrarna och utvärdera dess prestanda.
