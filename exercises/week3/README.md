# Övningsuppgifter vecka 3

## Python Databehandlingsövningar

## Pandas-övningar

1. Skapa en DataFrame från en dictionary av listor som innehåller information om 5 olika länder (namn, befolkning, yta, kontinent).

2. Ladda filen 'sample_data0.csv' och visa de sista 10 raderna.

3. Beräkna och visa medianlönen för varje avdelning.

4. Hitta den anställda med högst prestationspoäng i varje stad.

5. Skapa en ny kolumn 'Lön_per_År_Erfarenhet' genom att dividera 'Salary' med 'Years_Experience'. Hantera eventuella division-med-noll-fel.

6. Använd funktionen `pd.melt()` för att omforma DataFrame:n, och gör 'Salary' och 'Performance_Score' kolumnerna till variabler.

7. Pivota DataFrame:n för att visa genomsnittlig lön för varje kombination av Stad och Avdelning.

## NumPy-övningar

8. Skapa en 3x3 matris av slumpmässiga heltal mellan 1 och 10.

9. Utför elementvis multiplikation av två 4x4 matriser.

10. Använd NumPy för att lösa ett system av linjära ekvationer (Ax = b).

11. Generera en array med 1000 prover från en binomialfördelning med n=10 och p=0,5.

12. Skapa en 5x5 identitetsmatris och ersätt sedan dess diagonal med en anpassad array.

## Matplotlib- och Seaborn-övningar

13. Skapa ett staplat stapeldiagram som visar antalet anställda i varje Experience_Category för varje Stad.

14. Generera ett par-plot med seaborn för de numeriska kolumnerna i DataFrame:n.

15. Skapa ett violindiagram som jämför fördelningen av Performance_Scores över olika Avdelningar.

16. Gör ett cirkeldiagram som visar andelen anställda i varje Avdelning.

17. Skapa en 2x2 subplot med olika typer av diagram (linje, spridning, stapel och histogram) med data från DataFrame:n.

## Kombinerade övningar

18. Använd Pandas för att beräkna korrelationsmatrisen för numeriska kolumner, använd sedan seaborn för att skapa en heatmap av denna matris.

19. Skapa en NumPy-array med slumpmässiga tal, använd sedan Pandas för att skapa en DataFrame från denna array och Matplotlib för att plotta ett histogram av värdena.

20. Använd Pandas för att gruppera data efter 'City' och 'Department', beräkna genomsnittlig 'Salary' för varje grupp, använd sedan dessa data för att skapa ett grupperat stapeldiagram med Matplotlib.

## Scikit-learn-övningar

21. Använd scikit-learn:s `train_test_split`-funktion för att dela upp data i tränings- och testuppsättningar. Använd 'Salary' som målvariabel och 'Age', 'Years_Experience' och 'Performance_Score' som egenskaper.

22. Träna en enkel linjär regressionsmodell med scikit-learn för att förutsäga 'Salary' baserat på 'Years_Experience'. Plotta regressionslinjen tillsammans med ett spridningsdiagram av data.

Kom ihåg att importera de nödvändiga biblioteken (pandas, numpy, matplotlib.pyplot, seaborn och sklearn) innan du börjar med dessa övningar. Lycka till!

23. Vidare dataanalys och visualisering

Skapa ett program som använder pandas, numpy och matplotlib för att analysera och visualisera data från en CSV-fil med aktiemarknadsdata. Programmet ska:

1. Läsa in data och förbehandla den (hantera saknade värden, etc.)
2. Beräkna rullande medelvärden och standardavvikelser
3. Identifiera trender och anomalier
4. Skapa visualiseringar som linjediagram, histogram och scatterplots
5. Spara resultaten i en ny CSV-fil och bilderna som PNG-filer

24. Implementera en enkel linjär regression

För mer erfarna programmerare, implementera en enkel linjär regressionsalgoritm från grunden:

1. Skapa en klass `LinearRegression` med metoder för att träna modellen och göra prediktioner.
2. Implementera minsta kvadratmetoden för att hitta den bästa passformen.
3. Använd numpy för matrisoperationer.
4. Testa din implementation på ett enkelt dataset och jämför resultaten med sklearn's LinearRegression.

25. Skriv ett program för väderanalys och -prognos.

Se beskrivning av det önskade programmet under exercises/week3/Weather Analyzer!
[exercises/week3/Weather Analyzer](https://github.com/CalleFreme/Pythonprogrammering-for-AI-utveckling-HT24/blob/main/exercises/week3/Weather%20Analyzer/README.md)
Titta på stockapp.py för inspiration.
