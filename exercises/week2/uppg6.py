'''
Uppgift 6: Skapa en enkel filhanterare

Skriv en klass `FileManager` med följande metoder:

* `read_file(filename)`: Läser innehållet i en fil och returnerar det som en sträng.
* `write_file(filename, content)`: Skriver innehållet till en fil.
* `append_file(filename, content)`: Lägger till innehåll i slutet av en befintlig fil.
* `delete_file(filename)`: Raderar en fil.
'''

import os

class FileManager:
    @staticmethod
    def read_file(filename):
        try:
            with open(filename, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."
    
    @staticmethod
    def write_file(filename, content):
        try:
            with open(filename, 'w') as file:
                file.write(content)
            return f"Successfully wrote to '{filename}'."
        except IOError:
            return f"Error: Could not write to '{filename}'."
    
    @staticmethod
    def append_file(filename, content):
        try:
            with open(filename, 'a') as file:
                file.write(content)
            return f"Successfully appended to '{filename}'."
        except IOError:
            return f"Error: Could not append to '{filename}'."
    
    @staticmethod
    def delete_file(filename):
        try:
            os.remove(filename)
            return f"Successfully deleted '{filename}'."
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."
        except PermissionError:
            return f"Error: No permission to delete '{filename}'."

# Testa FileManager class
fm = FileManager()
print(fm.write_file("sample_text_filemanager.txt", "Hello, World!"))
print(fm.read_file("sample_text_filemanager.txt"))
print(fm.append_file("sample_text_filemanager.txt", "\nHow are you?"))
print(fm.read_file("sample_text_filemanager.txt"))
print(fm.delete_file("sample_text_filemanager.txt"))
print(fm.write_file("sample_text_filemanager.txt", "Hello again, World!"))
print(fm.read_file("sample_text_filemanager.txt"))

# Kommentarer:
# Denna klass använder statiska metoder för filhantering. Detta eftersom vi bara skapar ett objekt av klassen.
# Vi använder 'with'-satser för att säkerställa att filer stängs korrekt automatiskt efter vi gjort det vi vill med filen.
# Felhantering implementeras för att hantera vanliga filoperationsfel.
# os.remove() används för att radera filer, vilket kräver import av os-modulen.