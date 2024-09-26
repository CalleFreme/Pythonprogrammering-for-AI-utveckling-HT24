# functional.py
# Använd smarta, rena funktioner för att lösa komplexa problem över datasekvenser/listor
# För varje funktion, så som map(), filter(), reduce(), lambda-funktioner, läs dokumentation på nätet t.ex. https://www.geeksforgeeks.org/python-map-function/.

from functools import reduce

# Map function
print("Map example:")
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"Original: {numbers}")
print(f"Squared: {squared}")

# Filter function
print("\nFilter example:")
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Original: {numbers}")
print(f"Even numbers: {even_numbers}")

# Reduce function
print("\nReduce example:")
sum_of_numbers = reduce(lambda x, y: x + y, numbers)
print(f"Original: {numbers}")
print(f"Sum: {sum_of_numbers}")

# List comprehension
print("\nList comprehension examples:")
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# Lambda functions
print("\nLambda function examples:")
multiply = lambda x, y: x * y
print(f"4 * 5 = {multiply(4, 5)}")

# Sorting with lambda
print("\nSorting with lambda:")
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(f"Sorted by second element: {sorted_pairs}")

# Combining functional concepts
print("\nCombining concepts:")
sentence = "The quick brown fox jumps over the lazy dog"
word_lengths = list(map(len, filter(lambda word: not word.islower(), sentence.split())))
print(f"Lengths of words starting with uppercase: {word_lengths}")