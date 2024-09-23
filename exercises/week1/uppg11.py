'''
Uppgift 11
Skriv en funktion som tar emot en lista med ord
och returnerar det längsta ordet samt dess längd
'''

def get_longest_word(word_list):
    if (word_list): # We check that the word_list is not empty
        longest_word = max(word_list, key=len)
        longest_length = len(longest_word)
        return longest_word, longest_length
    return "", 0

# If you want to use a default word list, uncomment next 2 rows
# default_word_list = ["shortword", "longerword", "thiswordis16long"]
# word, word_length = get_longest_word(default_word_list)

# If you want to enter your own list of words:
my_word_list = []
while True:
    new_word_input = input("Enter a new word, or enter nothing if you want to stop: ")
    if (new_word_input == ""):
        break
    else:
        my_word_list.append(new_word_input)

word, word_length = get_longest_word(my_word_list)

print(f"Longest word is: \n{word}\nWith length: {word_length}")