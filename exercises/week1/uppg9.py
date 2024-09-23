'''
Uppgift 9
Skriv en funktion som vänder (reverse) på en 
sträng om dess längd är en multipel av 4.
'''

# Definitioner av funktioner måste skrivas före de används/kallas på/callas.
def reverse_string(string_to_reverse):
    input_str_length = len(string_to_reverse)

    if (input_str_length % 4 == 0): # Vi kollar om strängens längd är jämnt delbar med 4
        reversed_string = string_to_reverse[::-1]   # Kopierar hela strängen, omvänd.
        return reversed_string
    
    return string_to_reverse

input_string = input("Enter a string: ")

print(reverse_string(input_string))