
# bookCopy1 = {"title":"The Fellopship of the Ring", "author":"J.R.R Tolkien", "year":1954, "genre":"fantasy", "borrowed":True}
# bookCopy2 = {"title":"To Kill a Mockingbird", "author":"Harper Lee", "year":1960, "genre":"fiction", "borrowed":False}

# library = [bookCopy1, bookCopy2]

# for book in library:
#     print(book["title"])

from functools import reduce
import itertools

class Book:
    def __init__(self, title, author, year, genre, borrowed, pages):
        # Set up attributes
        self.title = title
        self.author = author
        self.year = year
        self.genre = genre
        self.is_borrowed = borrowed
        self.pages = pages

    def __str__(self):
        return f"{self.title} by {self.author} ({self.year}). Borrowed: {self.is_borrowed}"

    def borrow(self):
        if not self.is_borrowed:
            self.is_borrowed = True
            return True
        else:
            # Already borrowed
            return False
        
    def return_book(self):
        self.is_borrowed = False


class Library:
    def __init__(self, books=[]):
        self.books = books
        self.name = "Kungliga Biblioteket"

    def add_book(self, book):
        self.books.append(book)

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

def main():
    library = Library()
    print(library.books)
    print(library.name)

    book1 = Book("To Kill a Mockingbird", "Harper Lee", 1960, "Fiction", False, 281)
    library.add_book(book1)
    library.add_book(Book("1984", "George Orwell", 1949, "Science Fiction", False, 328))
    library.add_book(Book("Pride and Prejudice", "Jane Austen", 1813, "Romance", True, 432))
    library.add_book(Book("The Catcher in the Rye", "J.D. Salinger", 1951, "Fiction", False, 234))

    print("All books:")
    list(map(print, library.list_books()))

    print("-------------")
    for book in library.books:
        print(book)

    book_to_borrow = library.find_book("1984")
    if book_to_borrow:
        if book_to_borrow.borrow():
            print(f"\nBorrowed: {book_to_borrow}")
        else:
            print(f"\n{book_to_borrow} is already borrowed.")

    print("\nAvailable books:")
    for book in library.available_books():
        print(book)

main()
