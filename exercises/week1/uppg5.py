'''
Uppgift 5
Skriv ett program som lägger till "ing" i slutet av en given sträng,
om strängen är kortare än 3 tecken ska den lämnas ofärndrad.
Expempel-input: "Python"
Förväntad output: "Pythoning"
'''

input_string = input("Enter a string: ")

if len(input_string) >= 3:
    print(input_string + "ing")
else:
    print(input_string)