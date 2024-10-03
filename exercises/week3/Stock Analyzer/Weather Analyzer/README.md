# Specifikation: Väderanalys och prognos-program

## Översikt

Skapa ett Python-program som analyserar historiska väderdata, visualiserar trender och gör enkla temperaturprognoser. Programmet ska visa hur man använder populära dataanalys-bibliotek som Pandas, NumPy, Matplotlib och Seaborn.

## Datakälla

Använd en CSV-fil med namnet 'weather_data.csv' som innehåller följande kolumner:

Datum
Temperatur (°C)
Luftfuktighet (%)
Lufttryck (hPa)
Vindhastighet (m/s)
Nederbörd (mm)

## Huvudfunktioner

### 1. Databearbetning

Läs in väderdata från CSV-filen
Hantera eventuella saknade värden
Beräkna genomsnittlig temperatur per månad

### 2. Datavisualisering

Skapa ett linjediagram som visar temperaturtrend över tid
Gör ett stapeldiagram för månatlig nederbörd
Visa ett spridningsdiagram för temperatur vs luftfuktighet
Skapa en värmekarta för korrelation mellan olika väderparametrar

### 3. Enkel prognos

Använd en enkel metod (t.ex. glidande medelvärde) för att förutsäga morgondagens temperatur
Visa prognosen jämfört med faktiska temperaturer

### 4. Rapportering

Generera en sammanfattande rapport med nyckelstatistik

## Utdata

Programmet ska generera:

Fyra visualiseringar i PNG-format (trend, nederbörd, spridning, korrelation)
En textfil med sammanfattande statistik och prognosprestanda

## Tips

Börja med att importera nödvändiga bibliotek och läsa in data
Ta en funktion i taget och testa den innan du går vidare
Använd inbyggda hjälpfunktioner (t.ex. help(funktion)) för att lära dig mer om specifika funktioner
Kommentera din kod för att förklara vad varje del gör
Var inte rädd för att experimentera och göra misstag - det är så man lär sig!

## Eventuella utökningar

Lägg till fler avancerade visualiseringar
Implementera en mer sofistikerad prognosmodell
Använd scikit-learn för att till exempel skapa en modell som förutspår morgondagens temperatur
Jämför väderdata från olika platser
