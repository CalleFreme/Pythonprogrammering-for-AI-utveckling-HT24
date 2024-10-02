## Uppgift 16: List comprehensions vs. for-loopar

'''
Skriv list comprehensions för att:

1. Skapa och skriv ut en lista med alla tal mellan 1 och 100 som är delbara med 3 eller 5.
2. Generera och skriv ut en lista med tupler (x, y) för alla x och y där 0 <= x < 5 och 0 <= y < 5.

Skriv sedan två for-loopar som gör samma sak!
'''

# 1. Tal delbara med 3 eller 5
# List comprehension
divisible_nums = [n for n in range(1, 101) if n % 3 == 0 or n % 5 == 0]
print("Delbara med 3 eller 5 (list comprehension):", divisible_nums)

# For-loop
divisible_nums_loop = []
for n in range(1, 101):
    if n % 3 == 0 or n % 5 == 0:
        divisible_nums_loop.append(n)
print("Delbara med 3 eller 5 (for-loop):", divisible_nums_loop)

# 2. Generera tupler (x, y)
# List comprehension
coordinates = [(x, y) for x in range(5) for y in range(5)]
print("Koordinater (list comprehension):", coordinates)

# For-loop
coordinates_loop = []
for x in range(5):
    for y in range(5):
        coordinates_loop.append((x, y))
print("Koordinater (for-loop):", coordinates_loop)

# Kommentarer:
# List comprehensions ger ofta mer koncis och läsbar kod för enkla operationer.
# For-loopar är mer flexibla och kan vara lättare att förstå för komplexa operationer.
# Notera hur nästlade loopar hanteras i list comprehension vs. for-loopar.