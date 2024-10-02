# 1. Skapar en textfil och skriver några rader till den.
with open("example.txt", "w", encoding="utf-8") as file:
    file.write("Hej, världen!\n")
    file.write("Detta är en exempelfil.\n")

# 2. Läser innehållet i filen och skriver ut det. 'r' = read/öppna.
with open("example.txt", "r", encoding="utf-8") as file:
    print(file.read())

# 3. Lägger till mer text i slutet av filen. 'a' = append/lägg till. 'w' = write/skriv.
with open("example.txt", "a", encoding="utf-8") as file:
    # Filen skapas om den inte finns.
    file.write("Detta är ytterligare en rad.\n")

# 4. Läser filen igen och visar det uppdaterade innehållet.
with open("example.txt", "r", encoding="utf-8") as file:
    print(file.read())