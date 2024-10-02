## Uppgift 11: Enkla list comprehensions

'''
Öva på list comprehensions med följande uppgifter.

1. Skapa en lista med kvadrater av talen 1 till 10.
2. Filtrera ut alla jämna tal från en given lista.
3. Skapa en lista med längden av varje ord i en given mening.
'''

# 1. Kvadrater av talen 1 till 10
squares = [x**2 for x in range(1, 11)]
print("Kvadrater:", squares)

# 2. Filtrera ut jämna tal
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
print("Jämna tal:", even_numbers)

# 3. Längden av varje ord i en mening
sentence = "Python är ett kraftfullt programmeringsspråk"
word_lengths = [len(word) for word in sentence.split()]
print("Ordlängder:", word_lengths)

# Kommentarer:
# List comprehensions ger en koncis och läsbar syntax för att skapa listor.
# De kan inkludera villkor (if-satser) för filtrering.
# split() används för att dela upp meningen i ord.