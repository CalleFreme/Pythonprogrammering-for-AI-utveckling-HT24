## Uppgift 20: Intro Numpy - Matrisoperationer

'''
Använd numpy för att:

1. Skapa två 3x3 matriser med slumpmässiga heltal.
2. Beräkna produkten av dessa matriser.
3. Beräkna determinanten för den resulterande matrisen.
'''

import numpy as np

# 1. Skapa två 3x3 matriser med slumpmässiga heltal
matrix1 = np.random.randint(1, 10, size=(3, 3))
matrix2 = np.random.randint(1, 10, size=(3, 3))

print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)

# 2. Beräkna produkten av dessa matriser
product_matrix = np.dot(matrix1, matrix2)
print("\nProdukt av matriserna:")
print(product_matrix)

# 3. Beräkna determinanten för den resulterande matrisen
determinant = np.linalg.det(product_matrix)
print(f"\nDeterminanten av produktmatrisen: {round(determinant, 2)}")

# Kommentar på svenska:
# np.random.randint() används för att skapa matriser med slumpmässiga heltal.
# np.dot() beräknar matrisprodukten.
# np.linalg.det() beräknar determinanten av en matris.