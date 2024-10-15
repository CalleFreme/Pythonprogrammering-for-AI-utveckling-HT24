# Vecka 4 - Maskininlärningsövningar med TensorFlow, Keras och Scikit-learn

Dessa övningar är till för att hjälpa dig lära dig och öva på att använda TensorFlow, Keras och Scikit-learn för olika maskininlärningsuppgifter. Vi börjar med enklare övningar och ökar gradvis komplexiteten.

## Förberedelser

Läs README-filen under lectures/week4 innan du börjar med dessa uppgifter.
Denna [Geeksforgeeks-länk](https://www.geeksforgeeks.org/python-ai/) är en av många bra resurser.

## TEORIFRÅGOR

### 1. Vad är AI, och hur relaterar det till maskininlärning?

**Svar:**
Artificiell Intelligens är det bredare konceptet av maskiner som kan utföra uppgifter på ett sätt som vi skulle betrakta som "smart" eller "intelligent". Maskininlärning är en delmängd av AI som fokuserar på maskiners förmåga att ta emot data och lära sig själva utan att vara explicit programmerade.

### 2. Kan du förklara skillnaden mellan övervakad (supervised) och oövervakad (unsupervised) inlärning?

**Svar:**
I övervakad inlärning tränas algoritmen på en märkt dataset, där varje input är parad med korrekt output. Målet är att lära sig en funktion som mappar input till output. I oövervakad inlärning ges algoritmen omärkt data och måste hitta mönster eller struktur på egen hand.

### 3. Vad menas med klassifieringsinlärning (classification) och regression (regression)?

**Svar:**
Klassificering och regression är två huvudtyper av övervakad inlärning:

- Klassificering handlar om att förutsäga en diskret kategori eller klass. Till exempel att avgöra om ett e-postmeddelande är spam eller inte, eller att identifiera vilken siffra en handskriven bild föreställer.
- Regression handlar om att förutsäga ett kontinuerligt numeriskt värde. Till exempel att förutsäga huspriser baserat på egenskaper som storlek och plats, eller att förutsäga en persons inkomst baserat på utbildning och erfarenhet.

### 4. Vad menas med features (features) och labels (labels)?

**Svar:**
I maskininlärning refererar dessa termer till delar av träningsdata:

- Features (egenskaper) är de indata-variabler som används för att göra förutsägelser. De representerar de egenskaper eller attribut som beskriver varje exempel i datasetet. Till exempel kan features för ett huspris-dataset inkludera husets storlek, antal rum, och plats.
- Labels (etiketter) är de önskade utdata eller målvariablerna som modellen försöker förutsäga. I ett övervakat inlärningsproblem är labels de korrekta svar som modellen tränas att förutsäga. Till exempel kan labels i ett huspris-dataset vara de faktiska priserna för husen.

### 5. Vad är några vanliga tillämpningar av AI i vardagen?

**Svar:**
Några vanliga tillämpningar är:

Röstassistenter (Siri, Alexa)
Rekommendationssystem (Netflix, Amazon)
Bildigenkänning (Face ID på smartphones)
Skräppostfilter i e-post
Autonoma fordon

### 6. Vad är djupinlärning (deep learning) och hur skiljer det sig från traditionell maskininlärning?

***Svar:**
Djupinlärning är en underkategori av maskininlärning som använder neurala nätverk med många lager (därav "djup"). Det kan automatiskt lära sig komplexa mönster i data utan manuell feature extraction (d.v.s. ). Traditionell maskininlärning kräver ofta mer manuell feature engineering och fungerar bättre med mindre datamängder.

### 7. Hur fungerar ett neuralt nätverk på en grundläggande nivå?

**Svar:**
Ett neuralt nätverk består av sammankopplade noder (neuroner) organiserade i lager. Varje nod tar emot input, applicerar en aktiverningsfunktion och skickar output till nästa lager. Nätverket lär sig genom att justera vikterna mellan noderna baserat på fel i dess prediktioner.

### 8. Vad är förstärkningsinlärning (reinforcement learning) och hur skiljer det sig från övervakad och oövervakad inlärning?

**Svar:**
Förstärkningsinlärning är en typ av maskininlärning där en agent lär sig att interagera med en miljö för att maximera en belöning. Till skillnad från övervakad inlärning finns det ingen fördefinierad "korrekt" output, och till skillnad från oövervakad inlärning får agenten feedback i form av belöningar eller straff, och skapar en strategi-policy med tiden.

### 9. Vad är några populära Python-bibliotek som används inom AI och ML?

**Svar:**

- TensorFlow och Keras för djupinlärning
- scikit-learn för traditionell maskininlärning
- NumPy för numerisk beräkning
- Pandas för datamanipulering och analys
- Matplotlib och Seaborn för datavisualisering

### 10. Vad är "overfitting" och hur kan det undvikas?

**Svar:**
Overfitting inträffar när en modell lär sig träningsdatan för väl och presterar dåligt på ny, osedd data. Det kan undvikas genom tekniker som regularisering, early stopping, och användning av mer träningsdata.

### 11. Hur kan vi implementera bildigenkänning med Python?

**Svar:**
Vi kan använda deep learning-bibliotek som TensorFlow eller PyTorch för att träna en Convolutional Neural Network (CNN) på ett stort dataset av märkta (labeled) bilder. För enklare uppgifter kan vi också använda förtränade modeller som tillhandahålls av dessa bibliotek.

### 12. Hur kan vi utföra naturlig språkbehandling (NLP) med Python?

**Svar:**
Vi kan använda bibliotek som NLTK eller spaCy för grundläggande NLP-uppgifter. För mer avancerade uppgifter kan vi använda deep learning-modeller som BERT eller GPT, implementerade i bibliotek som Transformers från Hugging Face.

### 13. Hur kan vi implementera senitment-analys med Python?

**Svar:**
Vi kan implementera sentiment-analys i Python genom att använda en kombination av naturlig språkbehandling (NLP) och maskininlärning. Här är en grundläggande approach:

- Använd NLTK eller spaCy för förbehandling av text (tokenisering, borttagning av stoppord, etc.).
- Konvertera text till numeriska vektorer med hjälp av tekniker som Bag of Words eller TF-IDF.
- Träna en maskininlärningsmodell (t.ex. Naive Bayes eller Support Vector Machine) på en märkt dataset av texter med kända sentiments.
- Använd den tränade modellen för att förutsäga sentiment för nya, osedda texter.

För mer avancerade resultat kan man använda djupa inlärningsmodeller som LSTM eller BERT, implementerade med TensorFlow eller PyTorch.

### 14. Vad innebär det att normalisera datan?

**Svar:**
Att normalisera data innebär att skala om numeriska värden till ett gemensamt intervall, vanligtvis mellan 0 och 1 eller -1 och 1. Detta görs av flera anledningar:

1. Det hjälper till att jämna ut skillnader i storleksordning mellan olika features, vilket kan förbättra modellens prestanda.
2. Det kan göra träningsprocessen snabbare och mer stabil för många maskininlärningsalgoritmer.
3. Det hjälper till att undvika att features med större numeriska värden dominerar över features med mindre värden.

Vanliga normaliseringsmetoder är Min-Max-skalning och Z-score normalisering (standardisering).

### 15. Vad innebär en sekventiell modell?

**Svar:**
En sekventiell modell, särskilt i kontexten av deep learning-bibliotek som Keras, är en linjär stack av lager där data flödar från ett lager till nästa i en rak sekvens. Denna typ av modell är enkel att förstå och implementera:

1. Varje lager har exakt en input-tensor och en output-tensor.
2. Lager läggs till ett i taget i en bestämd ordning.
3. Det är den enklaste typen av neural nätverksarkitektur, där information flödar framåt genom nätverket utan några förgreningar eller hopp.

Sekventiella modeller är välpassande för många grundläggande deep learning-uppgifter, men för mer komplexa arkitekturer (som modeller med flera inputs eller outputs) kan man behöva använda den mer flexibla funktionella API:n.

### 16. Vad innebär en K-means-modell?

**Svar:**
K-means är en populär klusteringsalgoritm inom oövervakad inlärning. Den försöker gruppera datapunkter i K antal fördefinierade kluster baserat på deras likhet. Här är huvuddragen:

1. "K" representerar antalet kluster som algoritmen ska hitta.
2. Varje kluster representeras av sin centroid (mittpunkt).
3. Algoritmen itererar genom att tilldela varje datapunkt till det närmaste klustret och sedan uppdatera klustercentroiderna.
4. Processen upprepas tills centroiderna stabiliseras eller ett maximalt antal iterationer nås.

K-means används ofta för att upptäcka naturliga grupperingar i data, segmentera kunder, eller för dimensionsreduktion. En utmaning med K-means är att bestämma det optimala värdet för K, vilket ofta kräver experimentering och domänkunskap.

### 17. Vad innebär det att kompilera en modell?

**Svar:**
Att kompilera en modell, särskilt i kontexten av deep learning-ramverk som TensorFlow och Keras, innebär att förbereda modellen för träning genom att specificera vissa kritiska komponenter:

1. Optimerare (Optimizer): Algoritmen som kommer att uppdatera modellens vikter baserat på förlusten. Exempel inkluderar Adam, SGD, och RMSprop.
2. Förlustfunktion (Loss function): Måttet som används för att kvantifiera hur väl modellen presterar. Exempel inkluderar mean squared error för regression och binary crossentropy för binär klassificering.
3. Metriker (Metrics): Ytterligare mått som används för att övervaka träningsprocessen och modellens prestanda, som accuracy för klassificeringsuppgifter.

Kompileringsteget konfigurerar modellen för träning genom att sätta upp de matematiska operationer som krävs för att beräkna förlust, uppdatera vikter, och mäta prestanda. Det är ett nödvändigt steg innan man kan börja träna modellen på data.

## KODFRÅGOR

### 1. Vad gör följande kod, och vilken typ av maskinlärning använder den?

```python
from sklearn.cluster import KMeans
import numpy as np

# Generera exempel-data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Skapa och träna modellen
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Skriv ut kluster-centroids och labels
print("Centroids:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
```

***Svar:**
Programmet implementerar K-means clustering, vilket är en oövervakad inlärningsalgoritm. Den genererar först ett exempel-dataset med punkter i 2D-space. Sedan skapas en KMeans-modell med två kluster och tränas på datan. Slutligen skrivs centroiderna (mittpunkterna) för varje kluster ut, samt labels som indikerar vilket kluster varje datapunkt tillhör.

### 2. Vad gör följande kod?

```python
import tensorflow as tf
import pandas as pd
import numpy as np

# Ladda data från en CSV-fil
data = pd.read_csv('temperature_data.csv')

# Förbehandla data
X = data['Temperature'].values
y = data['Energy_Consumption'].values

# Normalisera data
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)

# Skapa en sekventiell modell
model = tf.keras.Sequential([¨
    # Skapar ett första lager i modellen. Dense innebär "fully connected layer", d.v.s varje neuron/nod är kopplad till alla andra noder i föreående och nästa lager.
    # 64 indikerar antalet noder i lagret.
    # input_shape(1,) innebär att vi förväntas oss endimensionell data, t.ex. ett enda tal per data-exempel i datasetet.
    # activation='relu' innebär att vi använder oss av aktiveringsfunktionen ReLu (Rectified Linear Unit). Vanlig.
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)), 
    tf.keras.layers.Dense(32, activation='relu'),   # Andra lagret har 32 noder. Får samma input_shape som föregående lager.
    tf.keras.layers.Dense(1)    # Sista lagret består av en enda slutgiltig nod, från vilken vi får den slutgiltiga outputen som modellen ska producera.
])

# Kompilera modellen
model.compile(optimizer='adam', loss='mse') # Modellen använder sig av 'Adam'-opmtimeringsfunktion, och Mean-Squared Error som loss-funktion.

# Träna modellen
model.fit(X, y, epochs=100, batch_size=32, verbose=0)

# Gör en förutsägelse
new_temp = np.array([25])  # Ny temperatur
new_temp_normalized = (new_temp - np.mean(X)) / np.std(X)
prediction = model.predict(new_temp_normalized)

print(f"Förutsagd energiförbrukning för temperatur 25: {prediction[0][0]}")
```

***Svar:**
Programmet implementerar en enkel neural nätverksmodell med TensorFlow och Keras för att förutsäga energiförbrukning baserat på temperatur. Den läser in data från en CSV-fil, normaliserar datan, skapar en sekventiell modell med tre Dense-lager, tränar modellen på datan, och gör sedan en förutsägelse för en ny temperatur.

### 3. Vad gör följande kod?

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Ladda data från en CSV-fil
data = pd.read_csv('movie_reviews.csv')

# Förbehandla text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['review'])

sequences = tokenizer.texts_to_sequences(data['review'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post')

# Skapa modell
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Kompilera och träna modellen
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, data['sentiment'], epochs=10, validation_split=0.2)

# Testa modellen på en ny recension
new_review = ["This movie was great! I really enjoyed it."]
new_sequence = tokenizer.texts_to_sequences(new_review)
new_padded_sequence = pad_sequences(new_sequence, maxlen=100, truncating='post')

prediction = model.predict(new_padded_sequence)
print(f"Sentiment prediction: {prediction[0][0]}")
```

***Svar:**
Programmet implementerar en sentimentanalysmodell med TensorFlow och Keras för att klassificera filmrecensioner som positiva eller negativa. Den läser in data från en CSV-fil, tokeniserar och paddar texten, skapar en sekventiell modell med ett Embedding-lager följt av GlobalAveragePooling och Dense-lager, tränar modellen på datan, och gör sedan en förutsägelse för en ny filmrecension.

---

## ÖVNINGSUPPGIFTER

Innan du börjar med övningarna, se till att du har följande bibliotek installerade:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Övning 1: Linjär regression med Scikit-learn

**Data**: ~~Använd Boston Housing-datasetet från Scikit-learn.~~
Boston-datasetet togs bort från Scikit-learn fr.o.m. version 1.2 på grund av etiska och juridiska problem. Tack till Aleh som upptäckte detta!
Använd istället California Housing dataset från Scikit-learn.

**Uppgift**: Skapa en enkel linjär regressionsmodell för att förutsäga huspriser.

**Steg**:

1. Ladda California Housing-datasetet med `sklearn.datasets.fetch_california_housing()`.
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
