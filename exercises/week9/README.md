# Week 9 - Model Evaluation and Comparison

1. Vad är syftet med korsvalidering (cross-validation) i maskininlärning?
Svar: Korsvalidering hjälper till att bedöma modellens prestanda på olika datauppdelningar för att undvika överanpassning och få en mer tillförlitlig uppskattning av modellens generalisering.

2. Vilken metrik är mest lämplig för en obalanserad klassificeringsdataset och varför?
Svar: F1-poäng är lämplig för obalanserade data eftersom den balanserar precision och recall, vilket gör den mer robust för klasser med få instanser.

3. Förklara vad ett ROC-AUC-diagram representerar.
Svar: ROC-AUC-diagrammet visar modellens förmåga att skilja mellan klasser. AUC-värdet representerar sannolikheten att modellen korrekt skiljer mellan en positiv och en negativ klass.

4. Vad är skillnaden mellan MSE och MAE i regression?
Svar: MSE (Mean Squared Error) straffar större fel mer än MAE (Mean Absolute Error) eftersom MSE kvadrerar felvärdena. MAE mäter bara medelfelet utan att förstärka stora fel.

5. När skulle man använda Leave-One-Out Cross-Validation (LOOCV) istället för k-Fold Cross-Validation?
Svar: LOOCV används när man har en liten dataset eftersom det utnyttjar alla data för träning förutom en enda observation som används för testning i varje iteration.

6. Vad är skillnaden mellan precision och recall?
Svar: Precision mäter andelen korrekt förutsagda positiva fall bland alla förutsagda positiva fall, medan recall mäter andelen korrekt förutsagda positiva fall bland alla verkliga positiva fall.

7. Vad innebär begreppet "ensembling" och varför används det?
Svar: Ensembling kombinerar flera modeller för att förbättra prestanda och stabilitet jämfört med enskilda modeller. Detta kan minska modellernas osäkerhet och förbättra generalisering.

8. Hur kan man använda SHAP-värden för att tolka en modells beslut?
Svar: SHAP-värden ger ett mått på varje features bidrag till modellens förutsägelse för en viss instans, vilket gör det möjligt att förstå hur enskilda features påverkar besluten.

9. Vad är syftet med Grid Search och hur fungerar det?
Svar: Grid Search söker systematiskt igenom ett definierat område av hyperparametrar för att hitta den kombination som optimerar modellens prestanda, baserat på en viss metrisk, exempelvis noggrannhet.

10. Förklara vad en silhuettpoäng är och hur den används vid klustring.
Svar: Silhuettpoängen mäter hur lik en datapunkt är sin egen kluster jämfört med andra kluster. En hög poäng indikerar att punkten är väl anpassad till sitt kluster, vilket indikerar bra klustring.
