
# bookCopy1 = {"title":"The Fellopship of the Ring", "author":"J.R.R Tolkien", "year":1954, "genre":"fantasy", "borrowed":True}
# bookCopy2 = {"title":"To Kill a Mockingbird", "author":"Harper Lee", "year":1960, "genre":"fiction", "borrowed":False}

# library = [bookCopy1, bookCopy2]

# for book in library:
#     print(book["title"])

import json
from functools import reduce
import itertools
from datetime import datetime, timedelta

# Decorator för att logga metodanrop.
# https://realpython.com/primer-on-python-decorators/
def log_method_call(func):
    def wrapper(*args, **kwargs):
        print(f"Anropar metod: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class Book:
    # Klassvariabel för att räkna antalet böcker
    book_count = 0  # Adderas med 1 för varje bok-objekt som skapas

    def __init__(self, title, author, year, genre, borrowed, pages):
        # Set up attributes
        self.title = title
        self.author = author
        self.year = year
        self.genre = genre
        self.is_borrowed = borrowed
        self.pages = pages
        self.due_date = None # Datumet som boken ska returneras
        Book.book_count += 1

    def __str__(self):
        return f"{self.title} by {self.author} ({self.year}). Borrowed: {self.is_borrowed}"

    @log_method_call
    def borrow(self):        
        if not self.is_borrowed:
            self.is_borrowed = True
            return True
        else:
            # Already borrowed
            return False
    
    @log_method_call
    def return_book(self):
        self.is_borrowed = False

    @staticmethod
    def is_overdue(due_date):
        return due_date and datetime.now() > due_date

class Library:
    def __init__(self, name, books=[]):
        self.books = books
        self.users = {}
        self.name = "Kungliga Biblioteket"

    @log_method_call
    def add_book(self, book):
        self.books.append(book)

    @log_method_call
    def remove_book(self, book):
        self.books.remove(book)

    def find_book(self, title):
        return next((book for book in self.books if book.title.lower() == title.lower()), None)

    def list_books(self):
        return list(map(str, self.books))

    def available_books(self):
        return list(filter(lambda book: not book.is_borrowed, self.books))

    def get_total_pages(self):
        return reduce(lambda x, y: x + y, map(lambda book: book.pages, self.books), 0)

    def group_by_genre(self):
        return {genre: list(books) for genre, books in 
                itertools.groupby(sorted(self.books, key=lambda x: x.genre), key=lambda x: x.genre)}

    # Generator (yield) för att iterera över böcker som skall returneras
    def overdue_books(self):
        for book in self.books:
            if book.is_borrowed and Book.is_overdue(book.due_date):
                yield book

    # Laddar in bibliotekets boklånar-användare från textfil
    def load_users_from_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    name, user_id, user_type = line.strip().split(',')
                    if user_type.lower() == 'loaner':
                        self.users[user_id] = Loaner(name, user_id)
                    elif user_type.lower() == 'librarian':
                        department = "General"  # You might want to add department to the file
                        self.users[user_id] = Librarian(name, user_id, department)
            print(f"Användare laddade från {filename}")
        except FileNotFoundError:
            print(f"Filen {filename} hittades inte. Inga användare laddade.")
        except Exception as e:
            print(f"Ett fel uppstod vid inläsning av användare: {e}")

    def get_user(self, user_id):
        return self.users.get(user_id)

    # Filhantering: Spara biblioteket till en JSON-fil
    def save_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'name': self.name,
                'books': [{'title': b.title, 'author': b.author, 'year': b.year, 
                           'genre': b.genre, 'borrowed': b.is_borrowed, 'pages': b.pages,
                           'borrower_id': getattr(b, 'borrower_id', None)} 
                          for b in self.books]
            }, f, ensure_ascii=False, indent=4)

    # Filhantering: Ladda biblioteket från en JSON-fil
    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            books = []
            for book_data in data['books']:
                book = Book(book_data['title'], book_data['author'], book_data['year'],
                            book_data['genre'], book_data['borrowed'], book_data['pages'])
                if book_data['borrower_id']:
                    book.borrower_id = book_data['borrower_id']
                books.append(book)
            library = cls(data['name'], books)
            return library
    

# Basklass för användare
class User:
    def __init__(self, name, user_id):
        self.name = name
        self.user_id = user_id  # Unikt för varje användare

    def __str__(self):
        return f"{self.name} (ID: {self.user_id})"

# Arv: Loaner (låntagare) ärver från User. Loaner är en "child" till User, som är en "parent" class. 
class Loaner(User):
    def __init__(self, name, user_id):
        super().__init__(name, user_id)
        self.borrowed_books = []

    def borrow_book(self, book):
        if book.borrow():
            self.borrowed_books.append(book)
            book.borrower_id = self.user_id
            return True
        return False

    def return_book(self, book):
        if book in self.borrowed_books:
            book.return_book()
            self.borrowed_books.remove(book)
            book.borrower_id = None
            return True
        return False

    def list_borrowed_books(self):
        return [str(book) for book in self.borrowed_books]

# Arv: Bibliotekarie ärver också från User
class Librarian(User):
    def __init__(self, name, user_id, department):
        super().__init__(name, user_id)
        self.department = department
        self.working_hours = {"Monday": [9,17], "Tuesday": [9,17], "Wednesday": [9,17], "Thursday": [9,17], "Friday": [9,16], "Saturday": [10,16], "Sunday": [11,16]}

    def add_book_to_library(self, library, book):
        library.add_book(book)

    def remove_book_from_library(self, library, book):
        library.remove_book(book)

def run_interactive_program(library):
    while True:
        print("\n--- Bibliotekssystem ---")
        print("1. Visa alla böcker")
        print("2. Låna en bok")
        print("3. Återlämna en bok")
        print("4. Visa lånade böcker")
        print("5. Avsluta")

        choice = input("Välj en åtgärd (1-5): ")

        if choice == '1':
            print("\nAlla böcker:")
            for book in library.list_books():
                print(book)
        elif choice == '2':
            user_id = input("Ange ditt användar-ID: ")
            user = library.get_user(user_id)
            if isinstance(user, Loaner):
                title = input("Ange titeln på boken du vill låna: ")
                book = library.find_book(title)
                if book:
                    if user.borrow_book(book):
                        print(f"Du har lånat: {book.title}")
                    else:
                        print("Boken är redan utlånad.")
                else:
                    print("Boken hittades inte.")
            else:
                print("Ogiltigt användar-ID eller användaren är inte en låntagare.")
        elif choice == '3':
            user_id = input("Ange ditt användar-ID: ")
            user = library.get_user(user_id)
            if isinstance(user, Loaner):
                title = input("Ange titeln på boken du vill återlämna: ")
                book = next((b for b in user.borrowed_books if b.title.lower() == title.lower()), None)
                if book:
                    user.return_book(book)
                    print(f"Du har återlämnat: {book.title}")
                else:
                    print("Du har inte lånat denna bok.")
            else:
                print("Ogiltigt användar-ID eller användaren är inte en låntagare.")
        elif choice == '4':
            user_id = input("Ange ditt användar-ID: ")
            user = library.get_user(user_id)
            if isinstance(user, Loaner):
                print("\nDina lånade böcker:")
                for book in user.list_borrowed_books():
                    print(book)
            else:
                print("Ogiltigt användar-ID eller användaren är inte en låntagare.")
        elif choice == '5':
            print("Tack för att du använder bibliotekssystemet!")
            break
        else:
            print("Ogiltigt val. Försök igen.")

def run_noninteractive_program(library):
        # Skapa en bibliotekarie och en låntagare om de inte redan finns
    if 'LIB001' not in library.users:
        librarian = Librarian("Anna Bibliotekarie", "LIB001", "Skönlitteratur")
        library.users['LIB001'] = librarian
    else:
        librarian = library.users['LIB001']

    if 'LOAN001' not in library.users:
        loaner = Loaner("Erik Låntagare", "LOAN001")
        library.users['LOAN001'] = loaner
    else:
        loaner = library.users['LOAN001']

    # Lägg till några böcker om biblioteket är tomt
    if not library.books:
        librarian.add_book_to_library(library, Book("Sagan om ringen", "J.R.R. Tolkien", 1954, "Fantasy", False, 423))
        librarian.add_book_to_library(library, Book("1984", "George Orwell", 1949, "Science Fiction", False, 328))

    print("\nAlla böcker:")
    for book in library.list_books():
        print(book)

    # Låna en bok
    book_to_borrow = library.find_book("1984")
    if book_to_borrow:
        if loaner.borrow_book(book_to_borrow):
            print(f"\n{loaner.name} lånade: {book_to_borrow}")
        else:
            print(f"\n{book_to_borrow.title} är redan utlånad.")

    print("\nTillgängliga böcker:")
    for book in library.available_books():
        print(book)

    print(f"\nTotalt antal sidor i biblioteket: {library.get_total_pages()}")

    print("\nBöcker grupperade efter genre:")
    for genre, books in library.group_by_genre().items():
        print(f"{genre}: {', '.join(book.title for book in books)}")

    print("\nFörfallna böcker:")
    for book in library.overdue_books():
        print(f"{book.title} är förfallen. Förfallodatum: {book.due_date}")

    print("\nErik Låntagares lånade böcker:")
    for book in loaner.list_borrowed_books():
        print(book)

def main():
    try:
        library = Library.load_from_file('library_data.json')
        print(f"Bibliotek laddat: {library.name}")
    except FileNotFoundError:
        library = Library("Kungliga Biblioteket")
        print(f"Nytt bibliotek skapat: {library.name}")

    library.load_users_from_file('library_users.txt')

    interactive_program = input("Vill du köra en interaktiv version av programmet (ja/nej)? ")
    if interactive_program.lower() == "ja":
        run_interactive_program(library)
    elif interactive_program.lower() == "nej":
        run_noninteractive_program(library)
    else:
        print("Felaktig input. Programmet avslutas.")

    # Spara biblioteket till fil
    library.save_to_file('library_data.json')
    print("\nBiblioteket har sparats till library_data.json")

    print(f"\nTotalt antal böcker skapade: {Book.book_count}")

if __name__ == "__main__":
    main()
