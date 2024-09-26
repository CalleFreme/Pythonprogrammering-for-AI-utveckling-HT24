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