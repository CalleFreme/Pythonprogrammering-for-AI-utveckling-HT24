'''
Uppgift 15
Skriv ett program som kontrollerar
om ett givet ord är ett palindrom 
(läses likadant framifrån som bakifrån, t.ex. "alla", "kajak").
Hur skulle du kunna skriva programmet med
'''

def is_palindrome(string):
    # Jämför strängen med dess reversed version
    return string == string[::-1]    # [::1] kopierar strängen rakt av, [::-1] kopierar den baklänges

original_string = input("Enter a string: ")

if is_palindrome(original_string):
    print(f"{original_string} is a paldindrome")
else:
    print(f"{original_string} is not a palindrome")

