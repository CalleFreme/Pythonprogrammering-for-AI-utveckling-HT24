## Uppgift 13: Funktionell programmering - Map och Filter

'''
Använd `map()` och `filter()` för att:

1. Skapa en lista med kvadrater av alla jämna tal i en given lista.
2. Filtrera ut alla primtal från en lista med tal.
'''

import math

# 1. Kvadrater av jämna tal
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print("Kvadrater av jämna tal:", even_squares)

# 2. Filtrera ut primtal
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

numbers = range(1, 20)
primes = list(filter(is_prime, numbers))
print("Primtal:", primes)

# Kommentarer:
# Vi kombinerar map() och filter() för att skapa kvadrater av jämna tal.
# För primtalsfiltrering använder vi en separat funktion is_prime() för tydlighetens skull.
# Notera användningen av math.sqrt() för effektiv primtalstest.