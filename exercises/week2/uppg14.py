## Uppgift 14: Reduce-funktion

'''
Importera och använd `functools.reduce()` för att:

1. Beräkna produkten av alla tal i en lista.
2. Hitta det största talet i en lista.
'''

from functools import reduce

# 1. Beräkna produkten av alla tal i en lista
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print("Produkt av alla tal:", product)

# 2. Hitta det största talet i en lista
max_number = reduce(lambda x, y: x if x > y else y, numbers)
print("Största talet:", max_number)

# Kommentar på svenska:
# reduce() applicerar en funktion av två argument kumulativt på en sekvens.
# Vi "reducerar" alla element i listan ner till _ett_ slutgiltigt resultat.
# För produktberäkning multiplicerar vi varje element med det ackumulerade/samlade värdet.
# För att hitta max-värdet jämför vi varje element med det hittills största värdet.