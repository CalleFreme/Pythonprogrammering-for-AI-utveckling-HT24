# Övningsuppgifter Vecka 2

Denna vecka handlar om Databearbetning, funktioner och OOP. Nedan finner du en lista med uppgifter. Vissa av dessa kommer vi gå igenom vid slutet av torsdagens lektion/handledningstillfälle. Vi uppmanar er att försöka lösa så många uppgifter som möjligt, vissa kommer kännas lättare, andra svårare. Fokusera på de uppgifter som känns tillräckligt utmanande för er nuvarande Python-färdighet, och samla på dig frågor på vägen. Chansen är stor att det du undrar över är det andra undrar eller borde undra över!

Fastnat? Bryt ner problemet/uppgiften, läs dokumentation, Googla och kom på så specifika frågor du kan att ställa till oss handledare. Det blir lättare för oss att hjälpa om vi vet vad du har försökt med hittills.

Lösningar till uppgifterna kommer finnas i denna mapp innan nästa vecka börjar.

Fail often, fail fast, som någon sa.

Genom att göra veckans uppgifter kommer du förstå:

* Mer Python
* Databearbetning i Python
* Funktionell programmering i Python, map(), reduce(), filter(), ..., lambda-funktioner,
* Klasser och objekt i Python, OOP
* Versionshantering och projektstruktur med Git

## Exempeluppgift 1 - `class Person`

Skapa en enkel klass `Person` med följande steg:

Definiera klassen `Person`.
Lägg till en konstruktor (`__init__`-metod) som tar emot `name` och `age` som parametrar.
Skapa en metod `introduce()` som returnerar en presentation av personen.
Skapa några instanser av `Person` och anropa `introduce()` metoden för var och en.

Lösningsförslag:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hej, jag heter {self.name} och är {self.age} år gammal."

# Skapa instanser och testa
person1 = Person("Anna", 30)
person2 = Person("Erik", 25)

print(person1.introduce())
print(person2.introduce())
```

## Exempeluppgift del 2 - Attribut och metoder för `Person`

Utöka Person-klassen från föregående uppgift:

1. Lägg till ett attribut `hobbies` som en lista i konstruktorn.
2. Skapa en metod `add_hobby(hobby)` för att lägga till en hobby.
3. Skapa en metod `get_hobbies()` som returnerar en sträng med alla hobbies.
4. Skriv över/overwritea `__str__` metoden för att ge en fin strängrepresentation av objektet, när man till exempel printar ett Person-objekt.

## Uppgift 1

Skriv en klass `Bankkonto`

* Den ska ha attributen `owner` och dess saldo/balance kommer initialt vara 0.
* Skapa metoder såsom `deposit(amount)` för att lägga till pengar till kontot samt `withdraw(amount)` för att ta ut pengar från kontot. Se också till att saldot ej kan bli negativ!
* Skapa en metod `display_balance()` för som printar kontots nuvarande saldo.

## Uppgift 2

Skriv en klass `Matte`
Den ska ha följande metoder

* add(a,b): Returnerar summan
* subtract(a,b): returnerar skillnaden
* divide(a,b): returnerar divisionen
* multiply(a,b): returnerar multiplikationen
* gcd(a,b): returnerar största gemensamma delare
* area_circle(r): returnerar arean av en cirkel
* circumference(d): returnerar omkretsen av en cirkel

## Uppgift 3: Grundläggande filhantering

Skriv ett program som gör följande:

1. Skapar en textfil och skriver några rader till den.
2. Läser innehållet i filen och skriver ut det.
3. Lägger till mer text i slutet av filen.
4. Läser filen igen och visar det uppdaterade innehållet.

Använd `with`-satser för att säkerställa att filen stängs korrekt.

Lösningsförslag:

```python
    # 1. Skapar en textfil och skriver några rader till den.
    with open("example.txt", "w") as file:
        file.write("Hej, världen!\n")
        file.write("Detta är en exempelfil.\n")

    # 2. Läser innehållet i filen och skriver ut det.
    with open("example.txt", "r") as file:
        print(file.read())

    # 3. Lägger till mer text i slutet av filen.
    with open("example.txt", "a") as file:
        file.write("Detta är ytterligare en rad.\n")

    # 4. Läser filen igen och visar det uppdaterade innehållet.
    with open("example.txt", "r") as file:
        print(file.read())
```

## Uppgift 4: Filläsning och ordräkning

Skapa en textfil (i samma mapp som ditt program). Skriv in eller kopiera in en text av valfri längd.
Läs in textfilen och använd ett dictionary för att räkna förekomsten av varje ord. Ignorera skiljetecken och konvertera alla ord till lowercase.

## Uppgift 5: Skapa en enkel kontaktbok

Implementera en klass `ContactBook` som använder ett dictionary för att lagra kontakter. Inkludera metoder för att lägga till, ta bort, uppdatera och visa kontakter.

## Uppgift 6: Skapa en enkel filhanterare

Skriv en klass `FileManager` med följande metoder:

* `read_file(filename)`: Läser innehållet i en fil och returnerar det som en sträng.
* `write_file(filename, content)`: Skriver innehållet till en fil.
* `append_file(filename, content)`: Lägger till innehåll i slutet av en befintlig fil.
* `delete_file(filename)`: Raderar en fil.

## Uppgift 7: Implementera en stack med en klass

Skapa en klass `Stack` som implementerar en stack (sista in, första ut) datastruktur med metoderna:

* `push(item)`: Lägger till ett element överst i stacken.
* `pop()`: Tar bort och returnerar det översta elementet i stacken.
* `peek()`: Returnerar det översta elementet utan att ta bort det.
* `is_empty()`: Returnerar True om stacken är tom, annars False.

## Uppgift 8: Skapa en enkel todo-lista

Implementera en klass `TodoList` med metoder för att:

* Lägga till uppgifter
* Markera uppgifter som slutförda
* Visa alla uppgifter
* Visa endast oavslutade uppgifter

## Uppgift 9: Intro till Arv och polymorfism

Skapa en hierarki av djurklasser:

1. Börja med en basklass `Animal` med attributen `name` och `sound`.
2. Skapa subklasser `Dog`, `Cat`, och `Cow` som ärver från `Animal`.
3. Överskugga `make_sound()` metoden i varje subklass för att returnera djurets specifika ljud.
4. Skapa en funktion `animal_chorus(animals)` som tar en lista av djur och låter alla göra sitt ljud.

## Uppgift 10: Arvs och polyformism: Klass och subklasser för geometriska former (med kod-skelett/scaffolding)

Här är en början på en klass för geometriska former. Komplettera klassen med metoder och funktionalitet enligt kommentarerna:

```python
class GeometricShape:
    def __init__(self, name):
        self.name = name

    def area(self):
        # Implementera en metod som returnerar formens area
        pass

    def perimeter(self):
        # Implementera en metod som returnerar formens omkrets
        pass

    def __str__(self):
        # Returnera en sträng som beskriver formen
        pass

class Rectangle(GeometricShape):
    def __init__(self, width, height):
        # Implementera konstruktorn
        pass

    # Implementera area() och perimeter() för Rectangle

class Circle(GeometricShape):
    def __init__(self, radius):
        # Implementera konstruktorn
        pass

    # Implementera area() och perimeter() för Circle

# Skapa några instanser av Rectangle och Circle och testa dina metoder
```

## Uppgift 11: Enkla list comprehensions

Öva på list comprehensions med följande uppgifter.

1. Skapa en lista med kvadrater av talen 1 till 10.
2. Filtrera ut alla jämna tal från en given lista.
3. Skapa en lista med längden av varje ord i en given mening.

Pröva lösa på egen hand innan du kollar lösningsförslagen nedan.

Exempel:

```python
# 1. Kvadrater
squares = [x**2 for x in range(1, 11)]

# 2. Filtrera jämna tal
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]

# 3. Ordlängder
sentence = "Python är ett kraftfullt programmeringsspråk"
word_lengths = [len(word) for word in sentence.split()]
```

## Uppgift 12: Intro till funktionell programmering

Läs på lite om map() och filter(), gör sedan följande uppgifter:

1. Skriv en funktion `double(x)` som returnerar det dubbla värdet av x.
2. Använd `map()` för att applicera `double` på en lista av tal, d.v.s. varje tal i listan ska dubblas.
3. Skriv en funktion `is_even(x)` som returnerar True om x är jämnt, annars False.
4. Använd `filter()` för att filtrera ut jämna tal från en lista.

Lösningsförslag:

```python
def double(x):
    return x * 2

numbers = [1, 2, 3, 4, 5]
doubled = list(map(double, numbers))

def is_even(x):
    return x % 2 == 0

even_numbers = list(filter(is_even, numbers))
```

## Uppgift 13: Funktionell programmering - Map och Filter

Använd `map()` och `filter()` för att:

1. Skapa en lista med kvadrater av alla jämna tal i en given lista.
2. Filtrera ut alla primtal från en lista med tal.

## Uppgift 14: Reduce-funktion

Importera och använd `functools.reduce()` för att:

1. Beräkna produkten av alla tal i en lista.
2. Hitta det största talet i en lista.

## Uppgift 15: Enkla lambda-funktioner

Vi introducerar lambda-funktioner, så kallade "anonyma funktioner" med några enkla exempel:

1. Skriv en lambda-funktion som returnerar kvadraten av ett tal.
2. Använd `sorted()` med en lambda-funktion för att sortera en lista av tupler baserat på det andra elementet.
3. Använd `filter()` med en lambda-funktion för att filtrera ut negativa tal från en lista.

Ta reda på och förklara hur lambda-funktioner skiljer sig från vanliga funktioner och när de är användbara.

## Uppgift 16: List comprehensions vs. for-loopar

Skriv list comprehensions för att:

1. Skapa och skriv ut en lista med alla tal mellan 1 och 100 som är delbara med 3 eller 5.
2. Generera och skriv ut en lista med tupler (x, y) för alla x och y där 0 <= x < 5 och 0 <= y < 5.

Skriv sedan två for-loopar som gör samma sak!

## Uppgift 17: Introduktion till Exception handling (undantagshantering)

Vi villa bygga våra program så att kan hantera olika situationer, till exempel när vi får fel typ av input från användaren.
Låt säga att du har ett program som tar in två heltal från användaren och utför en division på dessa två tal,
då måste du se till att ditt program kan hantera när täljaren är 0. Vi vill aldrig ha division med 0.
Vi hanterar dessa undantagsfall, Exceptions med `try`- och `except`-satser i vår kod. I andra programmeringsspråk ser man `catch` istället för `try`.
Det finns olika typer av Exceptions, så som `TypeError`, `ValueError`, `IndexError`, `KeyError`.
Ibland använder vi alla typer av errors som en generell `Exception` istället för specifik typ av Error. Detta ger oss dock
eventuellt mindre detaljerad information om vad som gått fel.

Skriv ett program som demonstrerar grundläggande undantagshantering:

1. Be användaren mata in två tal.
2. Försök (`try`) att dividera det första talet med det andra.
3. Hantera `ZeroDivisionError` om användaren försöker dividera med noll. (`except`)
4. Hantera `ValueError` om användaren matar in något som inte är ett tal. (`except`)

## Uppgift 18: Intro Matplotlib - Visualisering

Skapa ett linjediagram som visar temperaturdata för en vecka.
Som "data" räcker det t.ex. att skapa en lista med sju floats, en för varje dags medeltemperatur. ex: [15.5, 16.0, 14.6, 11.9, 15.3, 16.2, 15.7]
Använd matplotlib för att:

1. Plotta temperaturerna.
2. Lägga till en titel och etiketter för x- och y-axlarna.
3. Anpassa linjefärg och stil.

## Uppgift 19: Enkel dataanalys med Pandas

I denna uppgift kan ni använda CSV-filen `sales_data.csv` som finns här i repot.
Skapa ett program som använder Pandas för att analysera en CSV-fil med försäljningsdata:

1. Läs in en CSV-fil med kolumner för datum, produkt och försäljningsbelopp.
2. Visa de första 5 raderna.
3. Beräkna total försäljning per produkt.
4. Beräkna genomsnittlig försäljning per månad.
5. Hitta den dag med högst total försäljning.
6. Hitta produkten med högst total försäljning.
7. Skapa ett enkelt linjediagram över försäljningen över tid med matplotlib.

Om du vill:
8. Programmet sparar diagrammet som en PNG-fil
5. Programmet skriver en sammanfattning av analysen till en ny textfil (du får bestämma vad analysen ska inkludera)

Använd klasser för att strukturera koden och inkludera felhantering för filoperationer.

## Uppgift 20: Intro Numpy - Matrisoperationer

Använd numpy för att:

1. Skapa två 3x3 matriser med slumpmässiga heltal.
2. Beräkna produkten av dessa matriser.
3. Beräkna determinanten för den resulterande matrisen.

## Uppgift 21: Medelsvår - Skapa en egen iterator

Implementera en klass `FibonacciIterator` som genererar Fibonacci-sekvensen:

1. Använd `__iter__()` och `__next__()` metoder.
2. Låt iteratorn generera sekvensen upp till ett specificerat maxvärde.
3. Hantera `StopIteration` när sekvensen är klar.

Detta introducerar konceptet med iteratorer och generatorer på ett praktiskt sätt.

## Uppgift 22: Intro decorators

Detta ger en kort introduktion till dekoratorer och deras användning.
Skapa en enkel decorator `timer` som mäter exekveringstiden för en funktion:

1. Implementera decorator `timer`.
2. Använd `time` modulen för att mäta tiden.
3. Applicera dekoratorn på några funktioner med olika exekveringstider.
4. Forska lite på nätet hur dekoratorer fungerar och deras användningsområden.

## Uppgift 23: Generators

Implementera en generator som producerar Fibonacci-sekvensen upp till ett givet antal termer.

## Uppgift 24: Avancerad - Textanalysverktyg

Skapa ett textanalysverktyg som kombinerar filhantering, OOP, och funktionell programmering. Det ska kunna:

1. Läsa in en textfil
2. Räkna ord, meningar och stycken
3. Identifiera de vanligaste orden och fraserna
4. Beräkna läsbarhetsindex (Googla)
5. Generera en rapport med resultaten
