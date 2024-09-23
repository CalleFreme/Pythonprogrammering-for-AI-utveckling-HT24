'''
Uppgift 7
Skriv ett program som tar en komma-separerad sekvens av ord och 
skriver ut de unika orden i alfabetisk ordning.
Exempel-input: "red, white, black, red, green, black"
Förväntad output: "black, green, red, white"
'''

input_string_sequence = input("Enter a sequence of comma-separated strings: ")

# Vi tar bort eventuell whitespace med replace, och delar sedan upp alla ord i strängen med split,
# och stoppar in dem i en lista. sequence_list blir alltså en lista med orden som fanns i strängen. 
sequence_list = input_string_sequence.replace(' ', '').split(",") # Vi splittar efter varje "," i strängen, och får på så vis strängen uppdelar ord-vis

# Vi konverterar listan till ett set (en "mängd"), vilket automatiskt tar bort dubletter av värden i listan
sequence_set = set(sequence_list)

print(sorted(sequence_set)) # Får alfabetisk ordning på elementen i setet genom att använda sorted()
