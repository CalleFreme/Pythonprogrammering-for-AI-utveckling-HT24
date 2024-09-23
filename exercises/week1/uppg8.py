'''
Uppgift 8
Skriv en funktion som konverterar en given sträng till 
versaler (uppercase) om den innehåller minst 2 versaler 
bland de 4 första tecknen.
'''

input_string = input("Enter a string: ")

first_four_chars = input_string[0:4]    # Vi tar ut de första fyra tecknen/elementen i strängen
upper_count = 0 # Vi vill räkna antalet versaler vi hittat. Från början 0.

for char in first_four_chars:
    if char.isupper():
        upper_count += 1
    else:
        continue    # continue gör att loopen går vidare till nästa tecken direkt.
    
    if upper_count >= 2:
        # När vi hittat 2 versaler kan vi sluta leta
        input_string = input_string.upper() # Gör om hela strängen till stora bokstäver/versaler.
        break       # break gör att vi bryter oss ut ur loopen.

print(input_string)
