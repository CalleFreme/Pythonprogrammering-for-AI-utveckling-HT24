# Vecka 8 - Unsupervised Learning

Ett framework inom machine learning som, till skillnad från supervised learning, lär sig algoritmer
mönster från *enbart* unlabeled data.
Det generella målet är att hitta mönster och relationer i datat utan någon tidigare information.
Algoritmerna försöker i regel klustra unlabeled data, genom att hitta "gömda" mönster eller data-grupperingar, strukturer, på egen hand.

Som att lägga ett pussel utan referensbild, där mönster och samband måste upptäckas.

Har vi tur hittar algoritmen mönster som inte hade framkommit av ett labeled dataset.

Unsupervised learning kommer i olika former beroende på typ av data, träning, algoritm och tillämpning.
Datat samlas ofta "in the wild", från stora källor på nätet, till exempel i form av stora texter som hittas med hjälp av *web crawling*, där filtreringen ofta är minimal.

Fördelen är hur relativt billigt det är att utgå från unlabeled data jämfört med att skapa och konstruera labeled dataset.

Vanliga algoritmer är *clustering algorithms* så som ***k-means***, *association rules* så som ***Apriori***, och *dimenensionality reduction*-tekniker så som ***principal component analysis (PCA)***, ***Singular value decomposition (SVD)***, ***autoencoders***.

I och med utbredningen av deep learning, så utförs majoriteten av large-scale unsupervised learning genom
att träna general-purpose neural networks med *gradient descent*, där träningsprocedurer är särskilt utformade för oövervakad inlärning.

Gradient descent fungerar som så:

1. Algoritmen avgör åt vilket håll "marken" lutar mest nedåt
2. Tar ett steg i den riktningen
3. Upprepa till vi inte har någon lutninng

Gradient descent innebär rent matematiskt att algoritmen:

1. Beräkngar "lutningen" (gradienten) av en förlustfunktion.
2. Justerar modellens parametetrar (vikter) i små steg för att minska förlusten
3. Storleken på stegen kallas "learning rate".

Om vi exempelvis tränar en clustering-modell, vill vi minimera avståndet mellan datapunkter inom samma kluster. Gradient descent hjälper oss hitta den bästa placeringen av klustercentrum genom att stegvis justera deras positioner.

Oftast finjusteras förtränade modeller bereonde på slutändamålet, tillämpningen.

Utmaningar: Evaluation, Interpretability, Overfitting, Data quality, Computational complexity.
Möjligheter: Ingen labeled data krävs, hitta gömda mönster, många olika uppgifter, utforskning av data

## Användingsområden

Unsupervised learning innebär förmågan att utforska data på ett nytt sätt. Detta låter aktörer och företag identifiera
mönster i stora mängder data snabbare än egen observation.

Vi kan använda unsupervised learning för att hitta kungsegement, upptäcka bedrägeri, rekommendationssystem, NLP, bildanalys.

Vanliga användingsområden för unsupervised learning är:

- **Nyheter**
    - Automatisk gruppering av nyhetsartiklar efter ämne
    - Identifiering av trender i nyhetsflöden
    - Upptäckt av relaterade nyhetshändelser
- **Computer vision**
- **Medicinska bilder**
    - Identifiering av avvikelser i röntgenbilder
    - Gruppering av liknande patientfall
    - Upptäckt av dolda mönster i medicinsk data
- **Upptäcka anomaliteter**
- **Kundprofilerering**
    - Segmentering av kunder baserat på köpbeteende
    - Identifiering av kundgrupper med liknande preferenser
    - Anpassning av marknadsföringsstrategier
- **Rekommendationssystem**

## Supervised vs. Unsupervised Learning

Supervised Learning: Classification + Regression

- Mål: Approximera en funktion som mappar input till output baserat på labeled data.
- Träffsäkerhet: Hög träffsäkerhet, pålitligt.
- Mindre komplext
- Känt antal klasser
- Givna output-värden

Unupervised Learning: Clustering + Association + Dimensionality reduction

- Mål: Konstruera en koncis representation av datat och generara intressant innehåll baserat på det.
- Träffsäkerhet: Lägre träffsäkerhet, mindre pålitligt.
- Mer komplexa beräkningar
- Okänt antal klasser
- Inga givna output-värden

## Algoritmer

### Clustering Algorithms

Clustering är en typ av *data mining*-teknik som grupperar omärkt data baserat på deras likheter och olikheter.
Dessa algoritmer används för att bearbeta och gruppera rå data baserat på strukturer eller mönster i informationen däri.

Clustering-algoritmer kan delas upp i olika typer: *exlusive, overlapping, hierarchical,* och *probabilistic*.

### Exclusive och Overlapping clustering

#### K-means clustering

Att en metod är *exclusive* innebär att vi utför s.k. "hard" clustering, där varje datapunkt tillhöras exakt ett kluster, varken mer eller mindre.

K-means clusteringn är en vanlig typ av exclusive clustering, där datat delas upp i ***k*** antal grupper.
Grupperingen för en datapunkt baseras på dess avstånd till varje grupps "centroid", mittpunkt.
Ju mer lik datapunkten är en centroid-datapunkt, desto större chans att hamna i den centroidens grupp.
Ett större k-värde indikerar fler, mindre grupper.

1. Välj K antal startpunkter/mittpunkter/centroider
2. Tilldela varje datapunkt till närmaste centroid
3. Beräkna nya centroid-positioner baserat på medelvärdet av alla punkter i klustret
4. Upprepa steg 2-3 tills centroiderna stabiliseras

K-means används ofta inom market segmentation, document clustering, image segmentation, och image compression.

**Overlapping clustering** skiljer sig genom att låta datapunkter tillhöra flera grupper med olika grader av "membership", d.v.s. hur pass *mycket* de tillhör varje grupp.
"Soft" eller "fuzzy" k-means clustering är en typ av overlapping clustering.

### Hierarchical Clustering

Hierchical clustering, eller *hierarchical cluster analysis (HCA)*, är algoritmer som kan delas upp
i två olika typer: *agglomerative* och *divisive*.

#### Agglomerative Clustering

En s.k. "bottoms-up approach", där datapunkter isoleras som separata grupper till en början. De slås sedan samman iterativt
baserat på likheter tills ett enda cluster har formats.

Följande metoder används ofta för att mäta likhet, "avstånd":

* Ward's linkage: Avståndet mellan två kluster definieras av den ökade summan av kvadraten efter sammanslagning. För skapa så "kompakta" kluster som möjligt.
1. Beräkna medelvärdet inom varje kluster
2. När två kluster ska slås ihop, beräknas hur mycket den totala "spridningen" ökar
3. Välj att slå ihop de kluster som ger minst ökning i total spridning
* Average linkage: Avståndet definieras av det genomsnittliga avståndet mellan två datapunkter i varje kluster.
* Complete (maximum) linkage: Avståndet definieras av det maximala avståndet mellan två datapunkter i varje kluster.
* Single (minimum) linkage: Avståndet definieras av det minimala avståndet mellan två datapunkter i varje kluster.

Dessa avstånd i sig beräknas oftast med det *euklidiska avståndet*, "euclidean distance". Även Manhattan distance förekommer.

#### Divisive Clustering

Divisive clustering kan ses som motsatsen till agglomerative clustering, med en "top-down approach".
Vi utgår från ett enda kluster, och bryter upp det till mindre kluster baserat på skillnader mellan datapunkter.
Divisive clustering används inte särskilt ofta.

### Probabilistic Clustering

Algoritmer som låter oss uppskatta densitets-problem i data.
Datapunkterna grupperas baserat på sannolikheten att de tillhör en viss fördelning.
*Gaussian Mixture Model (GMM)* är en av de vanligaste metoderna.

#### Gaussian Mixture Models (GMMs)

En s.k. "mixture model", vilken innebär att modellen utgörs av ett ospecifierat antal sannolikhetsfördelningar.
Vi använder GMMs primärt för att avgöra vilken typ av Gaussian, normal- eller sannolikhetsfördelning som en given datapunkt tillhör.
Om genomsnittsvärdet eller variansen är känd, kan vi härleda typ av fördelning datapunkten tillhör.
I GMMs är dessa värden inte kända, men vi antar att det går att hitta en underliggande variabel som skulle klustra värden på ett lämpligt sätt.

Tar hänsyn till osäkerhet i kluster-tillhörighet, kan hantera kluster av olika storlek och form.

### Några andra clustering-algoritmer

**Density-based Clustering (DBSCAN)**: Komplexa/ojämna fördelningar av datapunkter
**Mean-Shift Clustering** (Mode-seeking algorithm): Antalet kluster bestäms av algoritmen i förhållande till datat.
**Spectral Clustering**: Gruppera baserat på "connectivity", kanter mellan noder. Graf-teori.

### Association Rules

Ett vanligt sammanhang där unsupervised learning används är inom "market basket analysis", där företag försöker förstå
relationer mellan olika produkter bättre. Genom att förstå kunders konsumptionsbeteenden, kan de utveckla bättre "cross-selling" strategier och rekommendationssystem.
Exempel är Amazons "Customers who bought this item also bought" och Spotify's "Discover Weekly".

Ett antal algoritmer används, så som Apriori, Eclat, och FP-Growth, men Apriori-algoritemn är mest populär.

#### Apriori

Skapar regler genom att analysera frekventa varukombinationer, skapar ett hash-träd för effektivitet. Används för att driva moderna rekommendationssystem. Avgör sannolikheten för att konsumption av en produkt leder till konsumptionen av en annan.
Apriori använder sig av ett hash tree för att gå igenom olika items, så som produkter. Går igenom trädet bredden först (breadth-first).
Apriori - använder sig av ett tidigare kunskap om frekventa item sets.

### Dimensionality Reduction Techniques

Även om mer data ofta innebär mer träffsäkra resultat, kan det också påverka modellens prestanda negativt, i form av t.ex. overfitting.
Vidare kan mycket data också innebära svårigheter att visualisera datasetet.
Bra för att förbättra prestanda hos algoritmer och för att visualisera data.

Vi använder dimensionality reduction när antalet features, d.v.s dimensioner, är för högt i ett dataset. Metoden reducerar antalet data-inputs till
en hanterbar storlek, samtidigt som datasetets integritet bibehålls i så hög grad som möjligt.

Dimensionality reduction används därmed ofta i *preprocessing*-steget. Metoder inkluderar Principal Component Analysis (PCA), Singular value decomposition (SVD), och autoencoders.

#### Principal Component Analysis (PCA)

PCA-algoritmer används för att minska redundans i och komprimera dataset genom *feature extraction*.
Processen går ut på att föst hitta en första "linjär transformation" av datasetet, d.v.s en representation av datat som leder till ett antal
"principal components", vilket ska maximera variansen i datasetet.
Den första principal component är den "riktning" som leder till mest varians i datasetet.
Den andra principal component, gör samma sak, men är helt orelaterad till den första komponenten och leder till en riktning som är vinkelrät till den första.
Detta fortsätter beroende på antalet dimensioner, där nästa principal component är ortogonal till den av de tidigare components som har högst variance.

Som att ta ett flerdimensionellt foto och hitta den bästa vinkeln att visa det från.

1. Hittar riktningar där data varierar mest
2. Behåller de viktigaste riktningarna
3. Reducerar dimensioner medan viktig information bevaras

#### Singular value decomposition (SVD)

En typ av dimensionality reduction som använder matriser för att utföra uppgifter som PCA, så som
noise reduction och data-komprimering av t.ex. bild-filer.

#### Autoencoders

Kan ses som en digital komprimeringstjänst.

Autoencoders använder sig av neural networks för att komprimera data och sen återskapa en ny representation av datats ursprungliga input.
Steget mellan Input Layer och Hidden Layers kallas för "encoding", steget mellan Hidden Layers och Output Layer kallas "decoding".

## Semi-Supervised Learning

Kombinerar supervised och unsupervised learning, behöver endast att en liten del av träningsdatat är labeled.

## Self-Supervision

Ses som en typ av unsupervised learning av vissa forskare.
I likhet med unsupervised learning, lär sig algoritmerna från unlabeled data. Till skillnad från unsupervised learning, så sker
inte inlärning från datats inbyggda struktur.
I likhet med supervised learning, är målet att generera en classified output för en viss input.
Å andra sidan, behövs inte labeled input-output-par.

SSL är särskilt bra för speech recognition och text processing.

## Välja typ av unsupervised learning

Tänk på:

1. Datasetets storlek och komplexitet
2. Tillgängliga beräkningsresurser
3. Krav på tolkningsbarhet
4. Specifika domänbegränsningar