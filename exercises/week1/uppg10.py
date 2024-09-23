'''
Uppgift 10
Skriv en funktion som skapar en ny sträng bestående av 
4 kopior av de två sista tecken i en given sträng.
Exempel-input: "Python"
Förväntad output: "onononon"
'''


def repeat_last_two(original_string):
    if len(original_string) >= 2:
        last_two_chars = original_string[-2:]
        repeating_string = last_two_chars*4
        return repeating_string
    else:
        # We just return the string if it's too short (shorter than 2)
        return original_string

input_string = input("Enter a string: ")

print(repeat_last_two(input_string))