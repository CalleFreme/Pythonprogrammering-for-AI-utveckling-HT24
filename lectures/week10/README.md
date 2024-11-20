# Vecka 10 - Deep Learning

**AI -> ML -> DL**
Deep learning är en definierande hörnsten i modern AI, och används inom allt från bildigenkänning till språkmodeller och generativ AI.
DL erbjuder oändliga möjligheter men kräver omsorgsfull design, träning och optimering.

## Välkommen till framtiden!

Deep learning är en revolutionerande teknik som bland annat möjliggör:

- **Gen**erative **AI**
- **Large Language Models (LLMs)**
- **Chatbots**
- **Deep fakes**

## Bra källor och läsning:

[Wikipedia: Deep learning](https://en.wikipedia.org/wiki/Deep_learning)
[DeepLearning.AI](https://www.deeplearning.ai/courses/) (*Bör gå att ladda ner vissa kursers material/slides*)
[Lex Fridman, MIT: Deep Learning Basics](https://www.youtube.com/watch?v=O5xeyoRL95U)

Googla efter artikel "Attention is all you need".
Läs TensorFlow/PyTorch dokumentation.
HuggingFace transformers library.

## Vad är Deep Learning?

DL handlar om neurala nätverk.
Exceptionellt bra för classification, regression, representation learning.

Arkitekturerna har tagits inspiration från hjärnans uppbyggnad, där konstgjorda "neuroner" staplas i lager och tränas att bearbeta data.
Att vi säger "deep" learning syftar alltså till att vi använder flera lager av neuroner, vilket kan vara allt mellan tre till flera hundra eller tusen lager.
Precis som människohjärnan så är neural networks oförutsägbara, input ger inte alltid samma output.

Den interna inlärningsmodellen som används kan vara supervised, semi-supervised eller unsupervised.
Deep learning är till stor del det som låtit oss avancera inom unsupervised learning, där komplexa lager av neuroner kan själva hitta grupperingar och mönster i omärkt data.

## Vad är ett Neural Network?

**Komponenter:**

1. Inputs: Funktioner/attribut i data
2. Weights: Varje input multipliceras med en vikt som anger dess betydelse
3. Biases: Värde som justerar modellens känslighet
4. Activation Function: Transformerar summan av input + bias för att avgöra neuronens output

För varje neuralt nätverk har vi ett antal inputs, X1, X2, ..., Xn, för n inputs/features.
Varje input har ett värde 0 eller 1.
Varje input kan ses som en binär fråga, X1 kan t.ex. stå för "Klart väder?", d.v.s representerar en feature "clearWeather".
Svaret på frågan kan representeras som 1 för "Ja", eller 0 för "Nej".
Output vi är ute efter är huruvida vi ska gå ut eller inte.

En annan input X2 kan t.ex. vara "isNotDark", som också påverkar om vi ska gå utomhus eller inte.
En tredje input X3 kan t.ex. vara "isWarm".

Till varje input finns en kopplad weight-faktor, w1, w2, ..., wn. Detta värde "viktar" hur stor påverkan en input har.
Om w1 = 2 och w2 = 5, innebär det att värdet på input X2 har större påverkan. Weight avgör alltså hur viktig en input är.

Beslutet beror alltså på dessa tre faktorer/inputs, och huruvida vi ska gå ut eller inte måste också vägas mot en bias, ett tröskelvärde.
Värdet på denna bias kan bestämas till nånstans mellan best case och worst case-scenariot.

Best-case: X1 = 1, X2 = 1, X3 = 1, d.v.s det är klart väder, det är ljust ute, och det är varmt ute.
(1 * 2) + (1 * 5) + (1 * 1) = 8

Worst-case: X1 = 1, X2 = 0, X3 = 0,
(0 * 2) + (0 * 5) + (0 * 1) = 0

8 / 2 = 4

Bias skulle alltså t.ex. kunna sättas till 4, 4.5, eller 5, beroende på hur pass "övertygade" vi vill bli innan vi går utomhus.

Vi låter** X1 = 1, X2 = 0, X3 = 1**, d.v.s det är klart väder, det är mörkt ute, och det är varmt.
Vi låter** w1 = 2, w2 = 5, w3 = 1,** d.v.s vi bryr oss mest om att det är ljus ute, näst mest att det är klart väder.

Vi låter Y stå för "Vill jag gå utomhus?". Vi beräknar svaret till denna fråga med hjälp av vår activation function.
Y = w1*X1 + w2*X2 + w3*X3 - bias = 2*1 + 5*0 + 1*1 - 4 = 3 - 4 = -1
Y är mindre än 0, d.v.s vi ska inte ga utomhus.

### När blir det "Deep"?

När nätverket har *flera lager av neuroner* och använder backpropagation för att justera vikterna.

## Feedforward Neural Networks (FNNs)

- Datat går från Input till Output layer utan feedback.
- Enklaste typen av nätverk, används för grundläggande regression och klassificering

## Backpropagation

- En algoritm för att optimera vikterna genom att minska fel (loss)
- Steg:

1. *Forward Pass*: Utför en forward pass, resulterar i en output och en error/loss.
2. *Backward Pass*: Beräknar gradienter (hur loss förändras med vikterna)
3. *Uppdatera vikterna*: Baserat på värdet för "learning rate", subtrahera en del av viktens gradient från vikten.

De neuroner vars tidigare vikt lett till felaktiva outputs, leder alltså till att vikten för dessa neuroner minskas.

### Vanliga aktiveringsfunktioner

En neurons output beräknas med en activation function.

## ReLU (Rectified Linear Unit)

- Används ofta i Hidden layers. Dessa layers är fully connected, d.v.s ett Dense-lager i Python/Keras.
- Output: max(0, x)

## Sigmoid (output layer)

- Används för binär klassificering, i Output-lagret
- Output: O(x) = 1 / (1 + e^-x)

## Softmax (output layer)

- För multi-class klassificiering, i Output-lagret
- Normaliserar outputs till sanolikheter

**"Shallow" neural networks:**
Ett Hidden layer, med backprogagation

## Standard DL techniques

Initialization methods
Regularization: Reducerar överanpassning, med t.ex. L2, dropout.
Batch Normalization: Stabiliserar inlärning genom att normalisera inputs
Gradient Checking: Identifierar fel i backpropagation

Optimization algorithms: mini-batch gradient descent, Momentum, RMSProp, Adam; check convergence;
Learning rate decay scehduling to speed up models

### End-to-End learning

Lär allt i ett enda steg.

### Transfer learning

Återanvänd delar av en modell på nya uppgifter.

### Multi-Task learning

Lär flera relterade uppgifter samtidigt.

## Convolutional Neural Networks

- Används för bilddatat och datorseende (object detection, bildsegementering, neural styler transfer)
- Key Layers:

1. Convolutional Layer: Upptärcka mönster (t.ex. kanter)
2. Pooling Layer: Minskar dimensionerna, datastorleken

### Forskning och tillämpning

Vi kan lära oss mycket från forskningsartiklar, för att sedan använda olika tekniker och metoder i form av transfer learning i våra egna modeller.

### Deep Learning in Computer Vision

Vi kan använda CNNs för visuell detektering och igenkänning
*Object detection* och *image segmentation* är typiskt.

CNNs kan användas inom många olika domäner, forskningsfält och sammanhang.
Neural style transfer, för ansiktsigenkänning, och för att generera ny konst

## Sequence Models

- Hanterar sekventiell data (t.ex. text, ljud, video)
- Speech recognition, music synthesis, chatbots, machine translation, NLP, och mer

### Recurrent Neural Networks (RNNS)

- Minnesförmåga genom rekursion
- GRUs, LSTMS: Löser problemet med långvarigt beroende
- Bidirectional RNNs: Läser sekvensen i båda riktningar
Bra för temporal data.

## NLP, Attention och Transformers

### Attention Mechanism

Förstärk (augment) sequence-modeller med hjälp av en attention mechanism.
Detta är en algoritm som hjälper din model avgöra var dess uppmärksamhet ska fokuserar, givet an sekvens av input.

Kritiskt för modern speech recognition, maskinöversättning, audio data.

#### Transformers

Arkitekturer som GPT och BERT.
Transformer-arkitekturer är bra för Natural Language Processing och attention models.
NLP Tasks: NER; Question Answering; Text Summarization

## Generative AI

### Foundation Models (FM) och LLMs

Foundation Models tränas på massiva mängder data, och möjliggör:

LLM och textgenerering - modellerar språk. Givet en viss sekvens av ord, predicta vad nästa sekvens (svaret) ska vara.
Audio - Generera röstinspelningar baserat på gamla sekvenser av rösten., syntetiskt tal
Bild - Dall-E, Stable Diffusion
Video - Deep fakes

Nytt eller uppspytt? Kan modellerna verkligen vara kreativa? Diskussion!

## TensorFlow vs PyTorch