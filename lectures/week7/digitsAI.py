import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Ladda in MNIST datasetet
# MNIST är dataset med 28x28 grayscale-bilder av handskriva siffror 0-9
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Den ursprungliga datan består av bilder representerade som en 2D-array, d.v.s. en matris 28x28 pixlar stor.
# CNN-modeller förväntar sig input på formatet (batch_size, height, width, channels)
# Forma om datan att inkludera single channel dimension, krävs av Conv2D.
 # batch_size = -1 betyder att modellen får avgöra själv hur många bilder den ska ta in i taget
 # height = 28 för att bilden är 28 pixlar hög
 # width = 28 för att bilden är 28 pixlar bred
 # channels = 1, d.v.s. en grayscale-bild. Sätter till channels = 3 om RGB-bild.
X_train = X_train.reshape(-1, 28, 28, 1) 
X_test = X_test.reshape(-1, 28, 28, 1)

# Pixlarnas värden indikerar dess färg med ett värde mellan 0 och 255. 
# Vi vill normalisera datan, till pixel-värden emllan 0 och 1 istället.
X_train = X_train / 255.0 # Värden mellan 0 och 1 är lättare för AIn att jobba med.
X_test = X_test / 255.0

# De tio olika talen utgör varsin kategori.
# One-hot encoda labels (t.e.x siffran "3" blir [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Skapa en CNN-modell
# Sequential är den simplaste typen av neural network, där vi stackar lager på varandra i en sekvens. Data flyter genom nätverket i en enda riktning.
# Vi använder två convolutional layers tillsammans, där det första hitta linjer och kanter, 
# och det andra upptäcker hur dessa skapar mer komplexa former, som kurvor och cirklar
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Detta Convolutional layer hittar simpla mönster i bilden (kanter, linjer)
    # Conv2D delar upp bilden i 32 "förstoringsglass" som letar efter grundläggande mönster/detaljer i varsin 3x3 pixlar del av bilden.
    # Vi använder akterivingsfunktion 'ReLU', som helt enkelt gör att neuronerna ger output 0 om input är negativt, output 1 om input är positivt 
    
    MaxPooling2D((2, 2)), # MaxPooling2D gör bilden mindre men bibehåller viktiga detaljer
    # MaxPooling2D minskar antalet spatiala dimensioner genom att ta det största värdet i varje 2x2 region
    # Innebär att vi tittar på 2x2-regioner i bilden, hittar "basic patterns" i dessa, d.v.s köper ut de viktigaste detaljerna/mönstren i bilden

    Conv2D(64, (3, 3), activation='relu'), # Kombinerar simpla mönster till mer komplexa (kurvor, loopar).
    # Letar igen men med ökad precision, dubbelt så många "filter", d.v.s dubbelt så många förstoringsglass letar igenom bilden efter mönster/detaljer.
    
    MaxPooling2D((2, 2)), # Väljer ut de "starkaste detaljerna" i bilden igen

    # Konverterar de två-dimensionella upptäckta mönstren till en enkel lista
    Flatten(), # Som att rulla ut en poster till en lång pappersremsa
    # Dense layers förväntar sig 1D-input

    # Första beslutsfattande lagret. Fully connected Dense layer.
    # 128 neuroner är stort nog för att fånga komplexa mönster, litet nog för att begränsa risken för overfitting
    # Varje neuron tittar på alla hittade mönster i bilden, och lär sig olika kombinationer av features/detaljer/mönster, för att göra klassificeringsbeslutet
    Dense(128, activation='relu'), # Tar alla hittade mönster/detaljer och bestämmer siffran. 128 neuroner eftersom
    
    Dropout(0.5), # Overfitting, stänger av 50% av slumpade neuroner under träningen. Vi vill inte bero för mycket på enstaka features/detaljer för att avgöra siffror.
    # 0.5 är aggressivt nog för att begränsa overfitting, litet nog för att inte hindra inlärning.
    # Ungefär som att täcka över delar av bilden under träning, så nätverket måste lära sig fler sätt att känna igen siffror.
    # Bra om modellen t.ex. lär sig att upptäcka siffran "8" m.h.a fler karaktäristiska mönster för den siffran.

    # Sista beslutsfattande lagret.
    # Output layer. SoftMax-funktion làmpligt för klassificerings-problem.
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tränar modellen på många bilder, 5 gånger. En epoch går igenom ALL träningsdata. För varje epok förbättras förhoppningsvis nätverkets "förståelse".
# Vi delar upp träningsdata i grupper/batches om 32.
# Modellen gör predictions på alla 32 bilder, beräknar det genomsnittliga felet, och uppdaterar därefter neuronernas vikter, d.v.s nätverkets "förståelse"
# För varje bild kommer modellen att göra en gissning, kolla om gissningen var korrekt, justera sin "förståelse", gå till nästa bild
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

predictions = model.predict(X_test)

print("\nPrediction for the first image in the test set:", predictions[0].argmax())