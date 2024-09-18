# Övningsuppgifter Vecka 1

Denna vecka handlar om grundläggande Python. Nedan finner du en lista med uppgifter. Vissa av dessa kommer vi gå igenom vid slutet av torsdagens (onsdag för Grupp 1) lektion/handledningstillfälle. Vi uppmanar er att försöka lösa så många uppgifter som möjligt, vissa kommer kännas lättare, andra svårare. Fokusera på de uppgifter som känns tillräckligt utmanande för er nuvarande Python-färdighet, och samla på dig frågor på vägen. Chansen är stor att det du undrar över är det andra undrar eller borde undra över!

Fastnat? Läs dokumentation och/eller kom på så specifika frågor du kan att ställa till oss handledare. Det blir lättare för oss att hjälpa om vi vet vad du har försökt med hittills.

Lösningar till uppgifterna kommer finnas i denna mapp innan nästa vecka börjar.

Fail often, fail fast, som någon sa.

Genom att göra veckans uppgifter kommer du förstå:

* Grundläggande Python-syntax, indentering, kommentarer, typomvandling
* Variabler
* Pythons inbyggda print(), input(), len(), type(), range()
* Datatyper så som strängar, listor, sets, dictionaries (maps, hashmaps), booleans
* Sträng-formattering
* Sträng-operationer så som join(), split(), upper(), lower(), strip()
* List-operationer så som append(), index-hantering, sorted(),
* if-satser
* for- och while-loopar
* Operationer så som modulo, konkatenering
* funktioner (för den som vill, mer fokus på detta nästa vecka)

## Uppgift 1

Skriv ett program som emot en sträng som input och skriver ut längden på strängen.
Exempel-input: "thisIsAString"
Förväntad output: 13

## Uppgift 2

Skriv ett program som skriver ut frekvensen av tecken i en given sträng.
Exempel-input: "banana"
Förväntad output: {"b":1, "a":3, "n":2}

## Uppgift 3

Skriv ett program som för en given sträng skriver ut de två första och de två sista tecknen i strängen (på valfritt format)
Exempel-input: "banana"
Förväntad output: "ba na"

## Uppgift 4

Skriv ett program som tar två strängar som input och skapar EN ny sträng där de två första tecken i varje sträng bytts ut.
Exempel-input: "abc", "xyz"
Förväntad output: "xyc abz"

## Uppgift 5

Skriv ett program som lägger till "ing" i slutet av en given sträng, om strängen är kortare än 3 tecken ska den lämnas ofärndrad.
Expempel-input: "Python"
Förväntad output: "Pythoning"

## Uppgift 6

Skriv ett program som först tar bort all whitespace (mellanslag, tab (\t), newline(\n)), och sedan även tar bort alla tecken på ojämna indexvärden, från given sträng.
Exempel-input: "a string with spaces and a newline character\n"
Förväntad output: "atigihpcsnaelncaatr"

## Uppgift 7

Skriv ett program som tar en komma-separerad sekvens av ord och skriver ut de unika orden i alfabetisk ordning.
Exempel-input: "red, white, black, red, green, black"
Förväntad output: "black, green, red, white"

## Uppgift 8

Skriv en funktion som konverterar en given sträng till versaler (uppercase) om den innehåller minst 2 versaler bland de 4 första tecknen.

## Uppgift 9

Skriv en funktion som vänder (reverse) på en sträng om dess längd är en multipel av 4.

## Uppgift 10

Skriv en funktion som skapar en ny sträng bestående av 4 kopior av de två sista tecken i en given sträng.
Exempel-input: "Python"
Förväntad output: "onononon"

## Uppgift 11

Skriv en funktion som tar emot en lista med ord och returnerar det längsta ordet samt dess längd

## Uppgift 12

Skriv ett program som genererar en enkel multiplikationsmodell för tal 1-10. Hur snyggt kan du få tabellen? Läs på om sträng-formattering i Python.

## Uppgift 13

Skriv en funktion som beräknar fakulteten av ett givet tal

## Uppgift 14

Skapa ett enkelt gissningsspel där datorn väljer ett slumpmässigt tal mellan 1-100 (eller annat intervall), och låt användaren gissa tills de hittar rätt nummer.
För varje felaktig gissning berättar datorn om det rätta svaret är högre eller lägre än spelarens gissning.

## Uppgift 15

Skriv ett program som kontrollerar om ett givet ord är ett palindrom (läses likadant framifrån som bakifrån).

## Uppgift 16

Skriv ett python program som itererar mellan 1 och 50, 
*	om talet är delbart med 3 printar den "fizz"
*	om talet är delbart med 5 printar den "buzz", 
*	om talet är delbart med både 3 och 5 så printar den "FizzBuzz"
*	annars printar den bara ut talet
