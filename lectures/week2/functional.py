# functional.py
# Använd smarta, rena funktioner för att lösa komplexa problem över datasekvenser/listor
# För varje funktion, så som map(), filter(), reduce(), lambda-funktioner, läs dokumentation på nätet t.ex. https://www.geeksforgeeks.org/python-map-function/.

from functools import reduce

# Map function
# Map applicerar en funktion på varje element i en lista (eller annan iterable)
# och returnerar en ny lista med de returnerade värdena.
print("Map example:")
numbers = [1, 2, 3, 4, 5]
# Lambda-funktionen x: x**2 kvadrerar varje tal i listan numbers.
squared = list(map(lambda x: x**2, numbers))
print(f"Original: {numbers}")
print(f"Squared: {squared}")

# Filter function
# Filter skapar en ny lista med element som uppfyller ett visst villkor
print("\nFilter example:")
# Lambda-funktionen x: x % 2 == 0 behåller endast jämna tal
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Original: {numbers}")
print(f"Even numbers: {even_numbers}")

# Reduce function
# Reduce applicerar en funktion av två argument kumulativt på varje element i en lista
print("\nReduce example:")
# Denna lambda-funktion har två argument (x, y) som returnerar x + y, dvs. summerar alla tal i listan.
sum_of_numbers = reduce(lambda x, y: x + y, numbers)
print(f"Original: {numbers}")
print(f"Sum: {sum_of_numbers}")

#reduce kan användas för mycket mer än bara addition. Du kan använda den för att:
# Hitta det största värdet i en lista
# Multiplicera alla tal i en lista
# Bygga en sträng från en lista av ord

# List comprehension
# List comprehension är ett koncist sätt att skapa listor baserat på existerande listor
print("\nList comprehension examples:")
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# Här skapas en lista med kvadrater av jämna tal mellan 1 och 10
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# Lambda functions
# Lambda-funktioner är små anonyma funktioner som kan ha vilket antal argument som helst,
# men kan bara ha ett uttryck
print("\nLambda function examples:")
multiply = lambda x, y: x * y
print(f"4 * 5 = {multiply(4, 5)}")

# Sorting with lambda
# Lambda-funktioner kan användas som nycklar vid sortering
print("\nSorting with lambda:")
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
# Här sorteras paren baserat på det andra elementet i varje par (strängen)
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(f"Sorted by second element: {sorted_pairs}")

# Combining functional concepts
print("\nCombining concepts:")
sentence = "The quick brown fox jumps over the lazy dog"

# Denna nästa rad gör följande:
# 1. Delar upp meningen i ord
# 2. Filtrerar ut ord som inte börjar med en versal
# 3. Mappar len-funktionen över de kvarvarande orden för att få deras längder
word_lengths = list(map(len, filter(lambda word: not word.islower(), sentence.split())))
print(f"Lengths of words starting with uppercase: {word_lengths}")