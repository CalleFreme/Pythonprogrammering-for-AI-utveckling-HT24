## Uppgift 5: Skapa en enkel kontaktbok

'''
Implementera en klass `ContactBook` som använder ett dictionary för att lagra kontakter.
Inkludera metoder för att lägga till, ta bort, uppdatera och visa kontakter.
'''

class ContactBook:
    def __init__(self):
        self.contacts = {}  # När ContactBook-objektet skapas innehåller den en tom dictionary.
    
    def add_contact(self, name, phone, email):
        self.contacts[name] = {"phone": phone, "email": email}
    
    def remove_contact(self, name):
        if name in self.contacts:
            del self.contacts[name]
            return True
        return False
    
    def update_contact(self, name, phone=None, email=None):
        if name in self.contacts:
            if phone:
                self.contacts[name]["phone"] = phone
            if email:
                self.contacts[name]["email"] = email
            return True
        return False
    
    def display_contacts(self):
        for name, info in self.contacts.items():
            print(f"Name: {name}, Phone: {info['phone']}, Email: {info['email']}")

# Testa ContactBook class
book = ContactBook()
book.add_contact("Alice", "123-456-7890", "alice@email.com")
book.add_contact("Bob", "987-654-3210", "bob@email.com")
book.display_contacts()
book.update_contact("Alice", phone="111-222-3333")
book.remove_contact("Bob")
book.display_contacts()

# Kommentarer:
# Denna klass använder ett nested dictionary (dictionar(y/ies) inuti i en dictionary) för att lagra kontaktinformation.
# Vi implementerar CRUD-operationer (Create, Read, Update, Delete) för kontakterna.
# Notera hur vi hanterar uppdatering av kontakter med valfria parametrar.