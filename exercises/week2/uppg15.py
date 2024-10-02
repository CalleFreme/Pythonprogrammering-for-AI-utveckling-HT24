## Uppgift 15: Enkla lambda-funktioner

'''
Vi introducerar lambda-funktioner, så kallade "anonyma funktioner" med några enkla exempel:

1. Skriv en lambda-funktion som returnerar kvadraten av ett tal.
2. Använd `sorted()` med en lambda-funktion för att sortera en lista av tupler baserat på det andra elementet.
3. Använd `filter()` med en lambda-funktion för att filtrera ut negativa tal från en lista.

Ta reda på och förklara hur lambda-funktioner skiljer sig från vanliga funktioner och när de är användbara.
'''

# 1. Lambda-funktion för kvadrat
square = lambda x: x**2
print("Kvadrat av 5:", square(5))

# 2. Sortera tupler baserat på andra elementet
pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print("Sorterade par:", sorted_pairs)

# 3. Filtrera ut negativa tal
numbers = [-3, -2, -1, 0, 1, 2, 3]
positive_numbers = list(filter(lambda x: x >= 0, numbers))
print("Positiva tal:", positive_numbers)

# Kommentarer:
# Lambda-funktioner är anonyma funktioner som kan definieras "inline".
# De är användbara för enkla operationer och som argument till högre ordningens funktioner (higer order functions).
# Higher order functions är funktioner som tar in en funktion som argument, t.ex. map, filter, reduce.
# Exempel: filter-funktionen filtrerar listan 'number' (det andra argumentet) baserat på en lambda-funktin (det första argumentet).
# I sorteringsexemplet används lambda som nyckel för att välja sorteringskriterium.