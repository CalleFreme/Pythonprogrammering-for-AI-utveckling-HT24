## Uppgift 12: Intro till funktionell programmering

'''
Läs på lite om map() och filter(), gör sedan följande uppgifter:

1. Skriv en funktion `double(x)` som returnerar det dubbla värdet av x.
2. Använd `map()` för att applicera `double` på en lista av tal, d.v.s. varje tal i listan ska dubblas.
3. Skriv en funktion `is_even(x)` som returnerar True om x är jämnt, annars False.
4. Använd `filter()` för att filtrera ut jämna tal från en lista.
'''

# 1. & 2. Dubbla värden med map()
def double(x):
    return x * 2

numbers = [1, 2, 3, 4, 5]
doubled = list(map(double, numbers))
print("Dubblade värden:", doubled)

# 3. & 4. Filtrera jämna tal med filter()
def is_even(x):
    return x % 2 == 0

even_numbers = list(filter(is_even, numbers))
print("Jämna tal:", even_numbers)

# Alternativ lösning med lambda-funktioner
doubled_lambda = list(map(lambda x: x * 2, numbers))
even_numbers_lambda = list(filter(lambda x: x % 2 == 0, numbers))

# Kommentarer:
# map() applicerar en funktion på varje element i en itererbar (t.ex. en lista, string).
# filter() väljer ut element från en itererbar baserat på en funktion som returnerar True/False.
# Vi visar både namngivna funktioner och lambda-funktioner för jämförelse.