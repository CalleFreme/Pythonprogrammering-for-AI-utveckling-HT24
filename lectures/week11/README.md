# Vecka 11 - Natural Language Processing och Generative AI

Natural Language Processing (NLP) handlar om att förstå, analysera, generera och bearbeta mänskligt språk.
Med NLP kan vi förstå språk, översätta text, generera ny text och implementera chatbotar.

NLP är i sin natur kopplat till informations-inhämtning, kunskaps-represenation och "computational linguistics".

Datat består ofta av text-korpus, som bearbetas med hjälp av definierade regler, statistiska metoder, eller neural network-baserade metoder, d.v.s ML och DL.

(NLP ska inte förväxlas med neuro-linguistic programming)

**Bra länkar:**

* [Wikiepdia - NLP](https://en.wikipedia.org/wiki/Natural_language_processing)

## Historia

NLP tog sin början redan på 1950-talet, i och med Alan Turing och hans Turing-test.
Turing-testet inkluderar en uppgift som kräver förmågan att tolka och generara mänskligt/naturligt språk.

Symbol-baserad NLP tog stora kliv på 80-talet, när forskning fokuserades på områden så som "rule-based parsing" och semantics (med t.ex. Lesk-algoritmen).

Från 90-talet började statistiska metoder bli populära varefter framsteg kunde ses inom automatiserad maskin-översättning.
I och med 2000-talets utbredning av Internet har mängden tillgänglig data att träna på snabbt blivit enorm.
Med detta har förstås även fokuset på unsupervised learning och semi-supervised learning ökat.

**Nu? Deep learning gör modern NLP**

Traditionella regel-baserade approacher (computational linguistics) kombineras med statistiska metoder, machine learning och framförallt deep learning.

Modern forskning kring NLP är det som möjliggort de senaste framgångarna inom generativ AI i form av LLMs och bild-generering och förståelsen av mänskliga kravbilder.

I den moderna AI-världen står vi i en oändlig rymd av data, i form av inte minst Internet, och den mesta datan på Internet är text. Förmågan att analysera dessa enorma mängder data är helt essentiellt för de senaste AI-framstegen.

Förmågan att generera "nytt" innehåll, text eller visuella medier, innebär också en mer påtaglig etisk diskussion kring generativ AI, i likhet med de etiska diskussionspunkter som dykit upp kring computer vision och dess tillämpningar, samt inom autonoma robotar, fordon och vapen.

## Metoder: Symboliskt, statistiskt eller neurala nätverk

I den **symboliska typen** av NLP definierar vi regler för att tolka och manipulera symboler (t.ex. bokstäver och ord), i kombination med en dictionary som beskriver olika ords egenskaper eller betydelser. Innebär ofta att man definierar en s.k. "grammatik" och/eller regeler för att bryta ner ord till sin grundform (s.k. stemming).

Denna typ av metod har tappat en del av sin relevans sedan införandet av 2000-talets machine learning, och ännu mer sedan LLMs tog mer plats sedan 2020-talet.

Vi kan fortfarande ha använding av traditionella symboliska metoder, när t.ex. träningsdatat är begränsat, eller när vi vill utföra preprocessing i vår AI-pipeline, eller för postprocessing av AI-resultat eller för att plocka ut information från det.

**Statistiska metoders** avancemang från 90-talet bidrog till en period av s.k. "AI winter", som till viss del berodde på de regel-baserade metoderna begränsade effektivitet.
Introduktionen av besluts-träd (decision trees) och "hidden Markov models", som används inom Part Of Speech tagging, innebar en revolution i NLP.

**DL, Neurala nätverks-metoder** innebär att vi kommer förbi de statistiska metodernas begränsningar, som generellt kräver feature engineering. Neural networks är "state of the art" sedan ca 2015, med "semantic networks" och "word embeddings" som försöker fånga betydelsen (semantiken) hos ord.
Machine learning innebar också att alla tidigare metoder för maskin-översättning blev överflödiga.

**Modern NLP med deep learning** innefattar ofta:

- **Sequence-to-Sequence (seq2seq)**: Baserat på recurrent neural netowrks (RNNs). Används för översättning.
- **Transformer-modeller**: Innebär tokenization av språk och "self-attention" (att fånga beroenden och relationer mellan ord), för att förstå relationen mellan olika delar av ett språk. Transformer-modeller kan tränas med self-supervised learning på stora text-datamängder. *BERT* från Google gjorde ett stort avtryck, och används fortfarande i deras sök-motor.
- **LLMS / Autoregressiva modeller**: Transformer-modell som tränas för att förutspå nästa ord i en sekvens. Har möjliggjort moderna modeller så som GPT, Llama, Claude och Mistral.
- **Foundation-modeller**: Förbyggda och kurerade modeller som ger en grund att bygga robusta, specialiserade modeller på.

## Hur fungerar NLP?

Övergripande innebär NLP: *text processing*, *feature extraction*, *text analysis* och *modell-träning*.

Grunden för NLP bygger på några tradionella metoder:

1. *Tokenisering*:
    - Delar upp text i mindre enheter (t.ex. ord, fraser, tecken).
    - Meningen "Jag älskar NLP" blir till tokens: ['Jag', 'älskar', 'NLP'].
2. *Stop word removal*:
    - Ta bort vanliga ord (som "och" "men") för att fokusera på mer meningsfulla delar
3. *Stemming och lemmatization*:
    - Reducera ord till sin grundform (stemming) eller till en unik form (lemmatization).
    - Exempel: "springer", "sprang" -> "spring"
4. *Part-of-Speech Tagging (POS)*:
    - Identifierar ordens grammatiska roller, som substantiv eller verb.
5. *N-grams och phrase analysis*:
    - Identifierar sekvenser av ord för att fånga beroenden i text.
6. *Syntax- och semantikanalys*:
    - *Syntax*: Förstå meingsstrukturen
    - *Semantik*: Förstå betydelsen och kontexten

**Utmaningar i NLP** inkluderar biased data, feltolkningar, vokabulär som förändras, variationer i människors röster och språk.

## Transformerarkitekturen: Den moderna grunden

Transformers är en revolutionerande arkitektur som möjliggjort modern NLP och generativ AI.
Tekniken introducerades genom Attention is All You Need-artikeln (2017) och utgör grunden för modeller som GPT och BERT.

- *Attention Mechanism*: Modellen avgör vilka delar av inputen som är viktigast vid varje steg. Exempel: I meningen "Jag älskar NLP eftersom det är coolt", lägger modellen mer fokus på "coolt" för att förstå varför något är älskat.

*Egenskaper hos transformers*:

- *Skalbarhet*: Fungerar på enorma datamängder.
- *Parallellisering*: Kan bearbeta sekvenser snabbare än RNN/LSTM-modeller.
- *Pre-train & Fine-tune*: Förtränas på stora datamängder och kan enkelt anpassas (fine-tunas) för specifika uppgifter.

## Moderna framsteg i NLP och Generative AI

*Stora språkmodeller (LLMs)*:

- Modeller som GPT-4 och BERT är tränade på massiva mängder textdata
- De har "för-förståelse" av språk och kan tillämpas på olika uppgifter (översättning, sammanfattning, etc.).

*Prompt Engineering*:

- En nyckel-teknik där vi styr vad en modell ska göra genom att skriva effektiva instruktioner.
- Exempel: "Skriv en recension för en science fiction-bok"

*Self-Supervised Learning*:

- Modeller tränas utan mänskliga labels genom att prediktera dolda eller nästa ord i text.
- Exempel: Maskering av ord: "Jag [MASK] NLP" -> "Jag älskar NLP"

## Hur kan vi använda NLP?

NLP och generativ AI har enorma möjligheter för olika tillämpningar, både i forskning och praktiska projekt. Vi kan...

### Bygga chatbotar och assistenter

- Med LLMs som GPT kan vi skapa intelligenta chatbotar som förstår naturligt/mänskligt språk och ger mänskliga svar.
*Teknik:*
- Fine-tuning för domänspecifika konversationer.
- Kombinera NLP med API för att göra chatbotar interaktiva (t.ex. e-handel eller support)

### Textgenerering och kreativa tillämpningar

- Generera rapport, e-postmeddelanden, marknadsföringstexter
- Skapa textbaserat content så som manus, böcker, artiklar

### Maskinöversättning

- Google Translate och liknande system använder transformerbaserade modeller
- Skapa interna verktyg för att översätta företagsdokument eller domänsspeficika texter med mycket jargong. Hur ska t.ex. juridiska eller tekniska texter förstås?

### Sentimentanalys och textklassificering

- *Sentimentanalys:* Förstå kundfeedback, sociala mediedata
- *Textklassificering*: Sortera e-pot, sätta etikettera på textdata, kategorisera content

### Sammanfattning av text

- Komprimera stora mängder information till kortfattade och lättläsa sammanfattningar
- Exempel: "Hitta och sammanfattade de senaste forskningsrönen inom vårt företags sektor"

### Informationsutvinning

- Extrahera viktiga detaljer ur stora mängder ostrukturerad data
- Exempel: Hitta företagsnamn och kontaktuppgifter i CV, automatisera insamling av information kring olika processer i ditt företag

### Generativ AI för visuella projekt

- Kombinera NLP med bildgenerering (som DallE eller Stable Diffusion) för att skapa bildtexter eller generera konstverk.

## Hur börjar vi använda NLP och GenAI i egna projekt?

1. *Väl ett problem/domän/uppgift*
    - Vad vill du åstadkomma? (t.ex. chatbot, sentimentanalys, rapportgenerering)
2. *Välj verktyg, hitta resurser*
    - *Hugging Face Transformers*: För att implementera pre-trained models som GPT eller BERT.
    - *OpenAI API*: För att använda GPT-3 eller GPT-4 direkt i dina applikationer.
    - *SpaCy och NLTK*: För enklare NLP-uppgifter som tokenisering och POS-tagging.
3. *Bygg en prototyp*: Gör saker på en så enkel nivå som möjligt
4. *Fine-tuning och anpassning*
    - Ta reda på hur du kan träna en modell för ditt specifika användningsområde.
    - Kombinera stora språkmodeller med mindre, specifika dataset

## NLP Python setup

Några kod-rader som du kan ha användning av i ditt projekt!

### OpenAI's API

```python
pip install transformers torch pandas numpy
pip install openai langchain chromadb
pip install pytest black isort

from openai import OpenAI
import os

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str, model: str = "gpt-4") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            raise

    def generate_embeddings(self, text: str) -> list:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise
```

### Lokala modeller med Hugging Face

```python
from transformers import pipeline, AutoTokenizer, AutoModel

class LocalLLMService:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def generate_text(self, prompt: str) -> str:
        generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        return generator(prompt, max_length=100)[0]['generated_text']
```

## Modern data-insamling. Web scraping, Privacy, Rate Limiting

```python
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict

class ResponsibleScraper:
    def __init__(self, rate_limit: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Bot (educational purposes)'
        })
        self.rate_limit = rate_limit
        
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        results = []
        for url in urls:
            time.sleep(self.rate_limit)  # Respect rate limiting
            try:
                response = self.session.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                results.append({
                    'url': url,
                    'title': soup.title.string if soup.title else '',
                    'content': self._extract_main_content(soup)
                })
            except Exception as e:
                logging.warning(f"Failed to scrape {url}: {e}")
        return results
        
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        # Implement content extraction logic
        pass
```

## Preprocessing

```python
import pandas as pd
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text: str) -> str:
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        return ' '.join(tokens)
        
    def prepare_dataset(self, 
                       texts: List[str], 
                       labels: List[str] = None,
                       test_size: float = 0.2):
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if labels:
            return train_test_split(
                cleaned_texts, 
                labels,
                test_size=test_size, 
                random_state=42
            )
        return train_test_split(cleaned_texts, test_size=test_size, random_state=42)
```

## Chatbot med langchain

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ContextAwareBot:
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.llm = ChatOpenAI(temperature=0.7)
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, message: str) -> str:
        return self.conversation.predict(input=message)
    
    def get_conversation_history(self) -> str:
        return self.memory.buffer
```
