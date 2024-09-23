# intro_python.py

import random

random_int = random.randint(0,5)
print(random_int)

'''
multi-
line
comment
'''

# 1. Basic syntrax, kommentarer, printing
print("Hej, klassen!") # Printa i Python

# 2. Variabler och datatyper
x = 30               # int
y = 3.14            # float
name = "Calle"     # string
is_fun = True       # boolean

# 3. Type checking, typkonverterting
print(type(x))
z = str(x)
print(type(z))

a = "10"
b = int(a) + 1
b += 1      # Samma som b = b + 1
print(b)

# 4 string operations
print(len(name))
print(name.upper())
print(name.lower())
print("  string with spaces  ".strip())
my_sentence_string = "a string with multiple words"
string_list = my_sentence_string.split(" ")
print(string_list)

# 5 string formatting
print(f"My name is {name} and I'm {x} years old")
print("Pi is approximately {:.2f}".format(y))

# 6 Lists
fruits = ["apple", "banana", "cherry", "apple"]
fruits.append("date")
fruits.insert(1, "strawberry")
print(fruits[2])
print(fruits)
fruits_string = ", ".join(fruits)


# 7 Dictionaries
person_dict1 = { "name":"Alice", "age":30, "city":"New York" }
print(person_dict1["name"])
person_dict1["job"] = "Developer"
print(person_dict1["job"])

person_dict2 = { "name":"Calle", "age":31, "city":"Stockholm", "job":"Teacher" }

person_list = [] # =list()
person_list.append(person_dict1)
person_list.append(person_dict2)

print(person_list)

# 8 Sets
unique_numbers = {1, 2, 3, 4, 5, 5, 5}
print(unique_numbers)
unique_fruits = set(fruits)
print(unique_fruits)

# 9 Input från användare
#username_input = input("Please enter your username: ")
#print(f"You entered username: {username_input}")

# 10 Conditionals 
age = 20
if age >= 18:
    print("Du får gå på klubb")
elif age >= 13:
    print("Du är tonåring")
else:
    print("Du är ett barn")

#if username_input == "callefreme":
#    print("Hej det är ju calle!")

# 11 Loops
# For loops
for fruit in fruits:
    print(fruit)

# While loops
count = 0
while count < 5:    # Vi loopar så länge ("while") villkoret är sant. 
    print(count)
    count = count + 1

# Oändlig while-loop
count = 1
while True: # Detta villkor är alltid sant
    print(f"Count is now: {count}")
    count += 1
    if (count >= 50):
        break   # Bryter ur loopen

print("Range loop")
for i in range(5): # range(5) = [0,1,2,3,4]
    print(i)

# Använda listan eller strängens storlek/längd med range för att gå igenom listans index:
my_numbers = [5,2,1,25,61,21]
for i in range(len(my_numbers)):
    print(f"On index {i}, we have number: {my_numbers[i]}") # Vi använder indexet 'i' för att komma åt det värde/siffra som ligger på det indexet

# Vi kan också loopa igenom listan/strängens värden/element direkt:
for number in my_numbers:   # Variabeln 'number' kan egentligen heta vad som helst, men det hjälper om variabelnamnet är deskriptivt. 'number' ger en tydlig indikation på vad för typ av värde variabeln kommer vara.
    print(f"Number: {number}")

# Loopar med dictionaries
my_phonenumber_dict = {"0725123112":"Calle", "0738519472":"Anna", "0703126123":"Bertil"}
for key, item in my_phonenumber_dict.items():   # Loop-variablerna 'key' och 'item' hade kunnat heta t.ex. 'phone_number' och 'name' istället.
    print(f"Phone number {key} belongs to {item}")


# Functions
def greet(name):
    #print(f"Hello, {name}")
    return f"Hello, {name}!"

def squared(x):
    return x**2

greeting = greet("Calle")
print(greeting)

a = 3
a_squared = squared(a)
print(f"{a} squared is: {a_squared}")

# Enumerate
for index, number in enumerate(my_numbers): # enumare() parar ihop varje element i listan med ett index
    number_squared = squared(number)
    print(f"{number} on index {index} is: {number_squared}")
