'''
Uppgift 2
Skriv ett program som skriver ut frekvensen av tecken i en given sträng.
Exempel-input: "banana"
Förväntad output: {"b":1, "a":3, "n":2}
'''

input_string = input("Enter a string: ")

# Vi skapar en tom dictionary där vi kommer spara ner frekvenser för olika bokstäver i strängen
frequency_dict = {}

# Vi loopar/itererar igenom strängens alla tecken. För varje tecken kontrollerar vi om vi sett det tidigare eller inte.
for char in input_string:
    # Om vi inte redan sett tecknet, det vill säga,
    # om tecknet inte redan existerar som nyckel i vår dictionary,
    # så lägger vi till tecknet i dictionary, och sätter värdet för den nyckeln till 1.
    if (char not in frequency_dict.keys()):
        frequency_dict[char] = 1
    else:
        # Annars adderar vi 1 till den existerande frekvensen.
        frequency_dict[char] += 1

# Vi printar ut vår dictionary lite snyggt med en loop:
for char, frequency in frequency_dict.items(): # Såhär kan vi loopa/iterera igenom alla nyckel-värde-par i en dictionary
    print(f"{char}: #{frequency}")