# Föreläsning vecka 7 - Computer Vision

**Bra resurser:**
[Computer Vision](https://www.kaggle.com/learn/computer-vision)
[KerasCV](https://keras.io/keras_cv/)

## Vad är Computer Vision? (CV)

Computer vision, datorseende, är ett fält inom AI som fokuserar på att låta program tolka och bearbeta visuell data,
på ett sätt som försöker efterlika mänsklig syn.

Det finns många olika typer av computer vision. Olika modeller och tekniker används beroende på om vi vill:

- Identifiera vad en bild föreställer - **Image Clasification**
    - Äldre, traditionella metoder eller CNNs
- Identifiera, loaklisera, kategorisera flera objekt i en bild - **Object Detection/Image Recognition**
    - R-CNNs, Yolo, SSD, etc.
- Dela in bilder i segment - **Image Segmentation**
    - Semantic Segmentation vs. Instance Segmentation
    - U-Net, Mask R-CNN
- Skapa nya bilder - **Image Generation and Enhancement**
    - GANs, Super-Resolution
- med mera...

**Tillämpningar**:

- Sjukvård (diagnos av medicinska bilder)
- Autonoma fordon
- Rekommendationssystem inom retail
- m.fl.

## Hur representeras och bearbetas visuell data?

### Bilder som matriser

Digitala bilder representeras som matriser av pixel-värden. 

- Grayscale images: En matris med värden (0-255)
- Color images: **Tre** matriser, där matriserna representerar Red, Green och Blue channels,
    d.v.s varje pixels färg bestäms av värdena för Red, Green och Blue i just den pixeln.

### Pixel-intensitet och upplösning

- Resolution (upplösning): Antalet pixlar i bredd och höjd. Högre resolution innebär mer detaljerade bilder,
    men kräver mer beräkningskraft.
- Vikt av preprocessing (förbehandling):
    - Storleken av bilder behöver ofta standardiseras.
    - Värden utanför intervallet 0-255 behöver förmodligent as bort.
    - Vi behöver oftast normalisera färg-värdena från 0-255 till intervallet 0-1, lättare för modellen att jobba med


## Hur hittar vi features, mönster och egenskaper i bilderna?

### Traditionella Feature Extraction-tekniker

**Edge Detection**: Använder *Sobel* eller *Canny operators* för att markera kanter i bilden
**SIFT (Scale-Invariant Feature Transform)**: Upptäcker *key points* och deras *descriptors* i en bild,
    användbar för object recognition.
**HOG (Histogram of Oriented Gradients)**: Kodar 

De traditionella metoderna är ofta begränsade, då de i regel är "hand-crafted" för ett särskilt syfte, och
fungerar dåligt i generella situationer, särskilt när bilderna är komplexa.

Vi kan använda deep learning för att automatisera och förbättra feature extraction, här kommer CNNs och efterföljande
teknologier in i bilden.

## Convolutional Neural Networks (CNNs)

### Varför CNNs?

CNN har transformerat bildigenkänning genom att automatisera feature extraction.
De kan identifiera komplexa mönster och förhållanden i bilder genom flera nätverks-lager.

### Arkitekturen i CNNs

**Convolutional Layers**
Applicera s.k. *filters* (*kernels*) för att extracta features.

I Python, t.ex.:

```python
Conv2D(32, (3, 3), activation='relu', input_shape(28, 28, 1))
```

**Pooling Layers**
Minska spatial dimensions, gör modellen scale-invariant och kan utföra effektiva beräkningar.
I Python, t.ex.:

```python
MaxPooling2D((2, 2))
```

**Fully Connected Layers**
Output-layers som kopplar alla neuroner för att kombinera features och klassificera bilder

I Python, t.ex.:

```python
Dense(128, activation='relu'),
...
# Om vi har 10 möjiga kategorier, ska vi ha 10 neruoner i sista lagret
Dense(10, activation='softmax') 
```

Se ***digitsAI.py*** för exempel.

## Image Recognition Pipeline

### Data Preprocessing and Augmentation

**Image Resizing och Normalisering**: Garantera att all input-data har samma format
**Data Augmentation**: Skapa variationer i träningsbilderna (rotation, flipping, cropping) för att göra modellen mer robust

### Transfer Learning

- Använd förtränade (pre-trained) modeller (t.ex. VGG, ResNet, YOLO) som har tränats på stora dataset så som ImageNet och COCO.
- Finjustera dessa modeller till din behov för snabbhet och hög träffsäkerhet.

## Image Classification

## Object Detection

Till skillnad från image classification, där vi identifierar vad en hel bild föreställer för typ av *"main object"*, vill vi i ***object detection*** hitta och identifiera flera objekt i en bild.

Convolutional Neural Networks låter oss utföra effektiv *image classification*, ***Regional*** **Convolutinal Networks** låter oss utföra effektiv *object detection*.

Vanliga modeller för object detection är

- **YOLO** (You Only Live Once): Snabb, single-stage detector
Se exempel ***cocoAI.py***

- **SSD** (Single Shot Multibox Detector): Bra för real-time apps
- **Faster R-CNN**: Hög träffsäkerhet, bra för detaljerad analys

## Image Segmentation

### Semantic Segmentation

Varje pixel får en label (t.ex. 'background', 'object')

### Instance Segmentation

Hittar individuella instanser av objekt, (t.ex. göra skillnad på individuella människor i en folkmassa).

**Vanliga modeller** är

- **U-Net**: Populär inom medicinsk image segmentation
- **Mask R-CNN**: Tillför förmågan att hitta instancer av objekt, ger detaljerade gränser mellan objekt, på pixel-nivå.

## Image Generation and Enhancement

Handlar om att skapa nya biler eller att förbättra existerande, med tekniker så som *GANs* eller *Super-Resolution*.

### Bildgenerering

**Generative Adversarial Networks (GANs)** skapar realistiska bilder genom att lära sig karaktäristiska drag i verkliga bilder.

### Super-Resolution

**Super-Resolution GAN (SRGAN)** kan användar för att förbättra upplösningen hos en bild.

## Optical Character Recognition (OCR)

En gren inom CV där vi vill konvertera bilder innehållande text till text som är läsbar för en dator (t.e.x en vanlig textfil).

Populära verkyg är bl.a. Tesseract, Google Vision API, Microsoft Azures OCR service.

## Pose Estimation

Handlar om att upptäcka mänskliga ledpunkter (axel, armbåge, handled etc.)för att försöka approximera och förstå poser och rörelser för en människokropp.

Populära verktyg är OpenPose, och PoseNet. Identifierar "key points" och härleder sedan poser.

## 3D Scene Reconstruction

Handlar om att skapa en 3D-representation, modell av en miljö eller ett objekt, baserat på flera 2D-bilder tagna från olika vinklar.

Populära tekniker är Structure from Motion (SfM) och Multi-View Stereo (MVS).

## Visual Tracking

Handlar om att identifiera och följa objekt i en video över tid.

Populära modeller är SORT (Simple Online and Realtime Tracking), Deep SORT (använder CNN för bättre tracking performance), och Re3.

## Action and Activity Recognition

Handlar om att upptäcka actions och aktiviteter i en sekvens av bilder eller video. T.ex. "walking", "running", "dancing", genom att analysera rörelsemönster *över tid*.

Poplära modeller är Two-Stream CNNs, 3D CNNs, och LSTMs, eller Transformers.

## Image Restoration and Inpainting

Handlar om att återställa och förbättra skadade bilder. Inpainting handlar om att fylla in saknade eller korrumperade delar av en bild.

Populära modeller är, CNNs för denoising, GAN-baserade metoder för inpainting.

## Explainability i Image Recognition-modeller

I vissa domäner, så som sjukvård, krävs hög transparens för att förstå varför modeller göra vissa predictions.

**Grad-CAM** är en teknik som visualiserar de regioner i en bild som bidrar till vilka predictions modellen gör.
Användbart för debugging och för att förstå modellers biaser.
Se exempel i ***cocoExplained.py***

## Några tips

**Effektiv databehandling**:
Stora bilder kan kräva mycket minne; använd ***tf.data** pipelines för effektiv bearbetning

**Debugging och visualisering**:
Visualiseringsverktig så som **TensorBoard** kan användas för att monitorera träning, visualisera modell-lager, och hålla koll på metrics.

## Exempelprogram 1 - DigitsAI

## Exempelprogram 2 - CocoAI