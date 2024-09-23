'''
Uppgift 12
Skriv ett program som genererar en enkel multiplikationsmodell
för tal 1-10. Hur snyggt kan du få tabellen?
Läs på om sträng-formattering i Python.
'''

def generate_multiplication_table():
    print("    |", end="")
    for i in range(1, 11):
        print(f"{i:4}", end="")
    print("\n" + "-" * 45)

    for i in range(1, 11):
        print(f"{i:2} |", end="")
        for j in range(1, 11):
            print(f"{i*j:4}", end="")
        print()

generate_multiplication_table()