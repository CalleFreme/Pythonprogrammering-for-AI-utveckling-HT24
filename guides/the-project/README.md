# Välja och definiera projekt

Stora problem behöver ofta brytas ner i mindre delar för att kunna förstås och hanteras.
Genom att följa principen 'divide and conquer', d.v.s att dela upp något i hanterbara bitar, och hantera dem en i taget,
blir vägen fram till lösningen mer kontrollerad och fokuserad.

För att veta hur ens program ska struktureras, och vilka byggdelar det behöver, måste du först veta vad programmet ska kunna göra.
Vi kan komma på, utvärdera och implementera vårt projekt i stegen:
EXPLORE -> DEFINE -> EVALUATE
När vi har en intressant, genomförlig och väldefinierad projektidé kan vi
IMPLEMENT -> TEST + EVALUATE -> ITERATE (upprepa föregående steg)

Här kan du se en exempel-specifikation för ett projekt: [AI Book Recommender](https://github.com/CalleFreme/Pythonprogrammering-for-AI-utveckling-HT24/blob/main/example-projects/supervised-learning). Ditt egna dokument kan vara mer eller mindre detaljerat, men det kommunicera din idé tydligt.

## EXPLORE - Vad finns och vad vill jag?

Denna fas handlar om att hitta ett ämne som känns intressant och också är genomförligt.
Exakt hur "färdigt" ditt projekt är i slutändan är inte det viktigaste. Börja med att bygga något som fungerar på en enkel nivå.
Din idé måste inte vara helt unik. Ibland är huvudmålet att lära sig hur man bygger något, oavsett om det redan finns eller inte.

**Bra frågor att ställa sig själv:**

- Vad skulle jag vilja lära mig?
- Vad skulle jag vilja bygga?
- Vad skulle jag vilja veta?
- Vad tycker jag är intressant?
- Vad har jag redan kunskap om?
- Vilket problem skulle jag vilja lösa?
- Vilken tjänst skule jag vilja ge andra?
- Vad skulle jag vilja utforska?
- Vad skulle hjälpa mig eller andra?
- Vad vore kul?
- Vad skulle jag vara stolt över?
- Vad skulle motivera mig?
- Vad har andra gjort innan?

## DEFINE - Vad ska jag göra och hur ska jag göra det?

Denna fas handlar om att förstå ungefär hur ditt program ska fungera, vad du ska bygga.
Beskriv och definiera vad ditt program gör och vilket problem det löser, vilken funktion det har.
Beskriv vilken data som ska analyseras, vilken modell du ska använda på datat, och vad modellen ska lära sig från datat.
Följande rubriker är en bra utgångspunkt:

### Program- och problembeskrivning

Genom att analysera data X vill jag kunna identifiera/bestämma/förutsäga/kategorisera Y.
Programmet är (t.ex.) ett rekommendationssystem för...

### Features

1. Data collection
    - Lista med datakällor
2. Feature extraction, encoding
3. Model training (namnet på modellen)
4. User interface for inputting preferences

### Modell - Vilken modell ska vi använda, och varför?

* Vad har du för data?
* Hur stor är datamängden och hur komplex är datan?
* Vad är programmets mål? Vad ska den förutsäga?

### Data - Vilken data ska vi använda, och varifrån kommer den?

* Är datan labeled?
* Vilka features vill du titta på?
* Hur mycket data behövs för att göra bra predictions? Experimentera och fundera!
* Vi kanske inte vill använda dataset alls, och använder reinforcement learning?

### Requirements - vad behöver mitt program?


## EVALUATE - Är det möjligt, rimligt, bra, intressant, roligt?

* Vad har vi för presentanda?
* Vad påverkar prestandan?

## IMPLEMENT & ITERATE

Make it work,
Make it right,
Make it fast.

Börja med en liten del av programmet, definiera projektstruktur.
Små steg i taget, lägg en sten i taget.

## TEST - Funkar det? Funkar det på rätt sätt?
