## Uppgift 4: Filläsning och ordräkning

import string
import re

def count_words(filename):
    word_count = {}
    
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Läser in hela filen som en stor str och konverterar alla tecken till lowercase
        # Ta bort skiljetecken, men behåll svenska tecken
        text = re.sub(r'[^\w\s\åäöÅÄÖ]', '', text)
        words = text.split()    # Delar in strängen i ord. Tomt argument i split() delar in strängen text vid varje mellanslag, dvs ord-vis.
        
        for word in words:
            if word in word_count:
                word_count[word] += 1   # Vi har sett ordet förut, plussar dess frekvens
            else:
                word_count[word] = 1    # Vi har inte sett ordet förut, lägger till det nya ordet med frekvens 1
    
    return word_count

# Testa funktionen
filename = "sample_text_countwords.txt"  # Säkerställ att denna fil finns i samma mapp som ditt program
result = count_words(filename)
# Använder lambda-funktion för att sortera ordet efter frekvens.
for word, count in sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{word}: {count}")

# Kommentarer:
# Vi använder 'utf-8' encoding när vi öppnar filen för att hantera svenska tecken korrekt.
# Vi använder ett reguljärt uttryck för att ta bort skiljetecken men behålla svenska tecken (å, ä, ö, Å, Ä, Ö).
# Vi använder ett dictionary för att räkna förekomsten av varje ord.
# Slutligen sorterar vi resultatet för att visa de 10 vanligaste orden.