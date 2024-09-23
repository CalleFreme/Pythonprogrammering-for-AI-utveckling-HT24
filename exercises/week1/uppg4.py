'''
Uppgift 4
Skriv ett program som tar två strängar som input och skapar EN ny sträng där de två första tecken i varje sträng bytts ut.
Exempel-input: "abc", "xyz"
Förväntad output: "xyc abz"
'''

# Vi konverterar strängerna till listor för att lättare kunna modifiera dem
first_string = list(input("Enter the first string: "))
second_string = list(input("Enter the second string: "))

temp_character = first_string[0]    # Sparar ner första strängens första tecken i en "temporär" variabel
first_string[0] = second_string[0]  # Byter värdet på första strängens första tecken
second_string[0] = temp_character   # Byter värdet på andra strängens första tecken

temp_character = first_string[1]    # Upprepar processen för strängarnas andra (second) tecken
first_string[1] = second_string[1]
second_string[1] = temp_character

# Vi gör listorna till strängar igen med jon(), och lägger ihop till en enda sträng som sparas ner i variabeln 'combined_string'
combined_string = "".join(first_string) + " " + "".join(second_string)  
print(combined_string)
