'''
Uppgift 13
Skriv en funktion som beräknar fakulteten av ett givet tal
'''

def factorial(n):
    """
    Beräknar fakulteten av ett tal med hjälp av enkel rekursion.
    
    :param n: Det positiva heltalet vars fakultet ska beräknas
    :return: Fakulteten av n
    """
    # Basfallet: om n är 0 eller 1, returnera 1
    if n <= 1:
        return 1
    # Rekursivt fall: n * fakulteten av (n-1)
    else:
        return n * factorial(n - 1) # Rekursion: funktionen kallar på sig själv.

number = int(input("Skriv in ett heltal: ")) # Vi konverterar input-siffran till ett heltal direkt.
result = factorial(number)
print(f"Fakulteten av {number} är {result}")