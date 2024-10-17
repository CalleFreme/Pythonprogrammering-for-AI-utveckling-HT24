# 1. Introduction to the fantasy and sci-fi book recommendation system project
# 2. Data collection and preprocessing
# 3. Explaining and implementing the basic k-NN algorithm wiht limited number of features
# 4. A simple user interface
# 5. Test and discuss
# 6. Next: real data, API calls, using more features, optimization, real graphics

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

class BookData:
    def __init__(self):
        self.books_df = None
        self.feature_matrix = None

    def load_data(self):
        # Ska egentligen ladda in data från databas med API, eller en fil
        # Till vår prototyp börjar vi med att skapa eget litet dataset
        self.books_df = pd.DataFrame({
            'title': ['The Hobbit', 'Dune', 'Mistborn', 'Neuromancer', 'The Name of the Wind'],
            'author': ['J.R.R. Tolkien', 'Frank Herbert', 'Brandon Sanderson', 'William Gibson', 'Patrick Rothfuss'],
            'subgenre': ['High Fantasy', 'Sci-Fi', 'Epic Fantasy', 'Cyberpunk', 'Epic Fantasy'],
            'themes': ['Quest, Dragons', 'Politics, Space', 'Magic, Heists', 'AI, Hacking', 'Magic, Music'],
            'length': [310, 412, 541, 271, 662],
            'year': [1937, 1965, 2006, 1984, 2007],
            'rating': [4.7, 4.5, 4.6, 4.3, 4.8]
        })
        print("Data loaded successfully.")

    def preprocess_data(self):
        # Till en början fokuserar vi på subgenre, themes, och ratings som våra main features
        # Vi One-Hot encodar subgenre och themes, vilket skapar kategorier baserat på dessa features
        encoder = OneHotEncoder(sparse=False)
        
        subgenre_encoded = encoder.fit_transform(self.books_df[['subgenre']])
        subgenre_columns = encoder.get_feature_names(['subgenre'])
        
        themes_encoded = encoder.fit_transform(self.books_df['themes'].str.split(', ', expand=True))
        themes_columns = encoder.get_feature_names(['theme'])
        
        # Kombinera encoded features med rating
        # Vi representerar våra features med en matrix för att
        self.feature_matrix = np.hstack((
            subgenre_encoded,
            themes_encoded,
            self.books_df[['rating']].values
        ))
        
        print("Data preprocessed successfully.")

class UserData:
    def __init__(self):
        # Lagrar användarens preferenser. Implementeras i nästa iteration.
        self.user_preferences = {}

    def get_user_input(self):
        # Just nu frågar vi bara efter en favorit-bok.
        favorite_book = input("Enter the title of a fantasy book you enjoy: ")
        return favorite_book

class RecommendationModel:
    def __init__(self, book_data):
        self.book_data = book_data
        self.model = None

    def train_model(self):
        # Initiera och träna modellen
        self.model = NearestNeighbors(n_neighbors=3, metric='euclidean')
        self.model.fit(self.book_data.feature_matrix)
        print("Model trained successfully.")

    def get_recommendations(self, book_title):
        # Beräkna rekommendationer baserat på en bok du gillar

        # Hitta boken
        book_index = self.book_data.books_df[self.book_data.books_df['title'] == book_title].index
        if len(book_index) == 0:
            return "Book not found in our database."
        
        # Get the feature vector for the input book
        book_features = self.book_data.feature_matrix[book_index]
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(book_features)
        
        # Get recommended book titles (excluding the input book)
        recommended_books = self.book_data.books_df.iloc[indices[0][1:]]['title'].tolist()
        
        return recommended_books

class UserInterface:
    def __init__(self, recommendation_model, user_data):
        self.recommendation_model = recommendation_model
        self.user_data = user_data

    def run(self):
        print("Welcome to the Fantasy Book Recommender!")
        favorite_book = self.user_data.get_user_input()
        recommendations = self.recommendation_model.get_recommendations(favorite_book)
        
        print("\nBased on your favorite book, we recommend:")
        for book in recommendations:
            print(f"- {book}")

def main():
    # Initiera komponenter
    book_data = BookData()
    user_data = UserData()
    
    # Ladda in och preprocessa data
    book_data.load_data()
    book_data.preprocess_data()
    
    # Initiera och träna modellen
    recommendation_model = RecommendationModel(book_data)
    recommendation_model.train_model()
    
    # Kör user interface
    ui = UserInterface(recommendation_model, user_data)
    ui.run()

if __name__ == "__main__":
    main()