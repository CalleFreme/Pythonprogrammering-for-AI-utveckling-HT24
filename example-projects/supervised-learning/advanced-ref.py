import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Key Enhancements in this version:

# Expanded Dataset: We've increased the number of books to 20, providing a richer set of options.
# Improved Preprocessing: We're now using a more sophisticated preprocessing pipeline with StandardScaler for numerical features and OneHotEncoder for categorical features.
# User Preferences: Instead of asking for a favorite book, we now ask users to rate their interest in various subgenres and themes.
# Cosine Similarity: We've switched from k-NN to cosine similarity, which often works better for high-dimensional sparse data like ours.
# User Vector: We create a vector representing the user's preferences and compare it directly to our book features.

class BookData:
    def __init__(self):
        self.books_df = None
        self.feature_matrix = None
        self.preprocessor = None

    def load_data(self):
        self.books_df = pd.DataFrame({
            'title': [
                'The Hobbit', 'Dune', 'Mistborn', 'Neuromancer', 'The Name of the Wind',
                'The Way of Kings', 'A Game of Thrones', 'The Colour of Magic', 'Assassin\'s Apprentice',
                'The Eye of the World', 'Elantris', 'Snow Crash', 'The Blade Itself', 'Gardens of the Moon',
                'The Lies of Locke Lamora', 'The Final Empire', 'The Black Prism', 'Dragonflight', 'Ancillary Justice', 'The Fifth Season'
            ],
            'author': [
                'J.R.R. Tolkien', 'Frank Herbert', 'Brandon Sanderson', 'William Gibson', 'Patrick Rothfuss',
                'Brandon Sanderson', 'George R.R. Martin', 'Terry Pratchett', 'Robin Hobb',
                'Robert Jordan', 'Brandon Sanderson', 'Neal Stephenson', 'Joe Abercrombie', 'Steven Erikson',
                'Scott Lynch', 'Brandon Sanderson', 'Brent Weeks', 'Anne McCaffrey', 'Ann Leckie', 'N.K. Jemisin'
            ],
            'subgenre': [
                'High Fantasy', 'Sci-Fi', 'Epic Fantasy', 'Cyberpunk', 'Epic Fantasy',
                'Epic Fantasy', 'Epic Fantasy', 'Comic Fantasy', 'Epic Fantasy',
                'Epic Fantasy', 'Epic Fantasy', 'Cyberpunk', 'Grimdark Fantasy', 'Epic Fantasy',
                'Fantasy Heist', 'Epic Fantasy', 'Epic Fantasy', 'Sci-Fi Fantasy', 'Space Opera', 'Dystopian Fantasy'
            ],
            'themes': [
                'Quest, Dragons', 'Politics, Space', 'Magic, Heists', 'AI, Hacking', 'Magic, Music',
                'Magic, War', 'Politics, Dragons', 'Magic, Humor', 'Magic, Assassins',
                'Magic, Prophecy', 'Magic, Politics', 'Virtual Reality, Linguistics', 'War, Revenge', 'Magic, Empire',
                'Heists, Friendship', 'Magic, Revolution', 'Magic, Politics', 'Dragons, Telepathy', 'AI, Gender', 'Magic, Climate Change'
            ],
            'length': [
                310, 412, 541, 271, 662,
                1007, 694, 288, 392,
                814, 590, 470, 536, 666,
                499, 647, 640, 352, 386, 468
            ],
            'year': [
                1937, 1965, 2006, 1984, 2007,
                2010, 1996, 1983, 1995,
                1990, 2005, 1992, 2006, 1999,
                2006, 2006, 2010, 1968, 2013, 2015
            ],
            'rating': [
                4.7, 4.5, 4.6, 4.3, 4.8,
                4.7, 4.6, 4.2, 4.4,
                4.5, 4.3, 4.3, 4.4, 4.3,
                4.5, 4.6, 4.4, 4.2, 4.1, 4.5
            ]
        })
        print("Data loaded successfully.")

    def preprocess_data(self):
        # Preprocessing pipeline:
        categorical_features = ['subgenre', 'themes']
        numerical_features = ['length', 'year', 'rating']

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Encoding definierar katogorier på följande sätt:
        # Book    | High Fantasy | Sci-Fi | Epic Fantasy
        # ----------------------------------------------
        # Book 1  |      1       |   0    |      0
        # Book 2  |      0       |   1    |      0
        # Book 3  |      0       |   0    |      1

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit and transform the data
        self.feature_matrix = self.preprocessor.fit_transform(self.books_df)
        print("Data preprocessed successfully.")

class UserData:
    def __init__(self):
        self.user_preferences = {}

    def get_user_input(self):
        print("Please rate your interest in the following subgenres (1-5):")
        subgenres = ['High Fantasy', 'Sci-Fi', 'Epic Fantasy', 'Cyberpunk', 'Comic Fantasy', 'Grimdark Fantasy']
        for subgenre in subgenres:
            while True:
                try:
                    rating = int(input(f"{subgenre}: "))
                    if 1 <= rating <= 5:
                        self.user_preferences[subgenre] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")

        print("\nPlease rate your interest in the following themes (1-5):")
        themes = ['Magic', 'Dragons', 'Politics', 'War', 'Heists', 'AI']
        for theme in themes:
            while True:
                try:
                    rating = int(input(f"{theme}: "))
                    if 1 <= rating <= 5:
                        self.user_preferences[theme] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")

class RecommendationModel:
    def __init__(self, book_data):
        self.book_data = book_data
        self.model = None

    def train_model(self):
        # Use cosine similarity instead of Euclidean distance
        self.model = cosine_similarity(self.book_data.feature_matrix)
        print("Model trained successfully.")

    def get_recommendations(self, user_preferences):
        # Create a user vector based on their preferences
        user_vector = np.zeros(self.book_data.feature_matrix.shape[1])
        for subgenre, rating in user_preferences.items():
            if subgenre in self.book_data.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(['subgenre']):
                idx = np.where(self.book_data.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(['subgenre']) == subgenre)[0]
                user_vector[idx] = rating
        for theme, rating in user_preferences.items():
            if theme in self.book_data.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(['themes']):
                idx = np.where(self.book_data.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(['themes']) == theme)[0]
                user_vector[idx] = rating

        # Calculate similarity between user vector and all books
        user_similarity = cosine_similarity(user_vector.reshape(1, -1), self.book_data.feature_matrix)

        # Get top 5 most similar books
        top_indices = user_similarity.argsort()[0][-5:][::-1]
        recommended_books = self.book_data.books_df.iloc[top_indices]['title'].tolist()

        return recommended_books

class UserInterface:
    def __init__(self, recommendation_model, user_data):
        self.recommendation_model = recommendation_model
        self.user_data = user_data

    def run(self):
        print("Welcome to the Advanced Fantasy Book Recommender!")
        self.user_data.get_user_input()
        recommendations = self.recommendation_model.get_recommendations(self.user_data.user_preferences)

        print("\nBased on your preferences, we recommend:")
        for i, book in enumerate(recommendations, 1):
            print(f"{i}. {book}")

def main():
    # Initialize components
    book_data = BookData()
    user_data = UserData()

    # Load and preprocess data
    book_data.load_data()
    book_data.preprocess_data()

    # Initialize and train the model
    recommendation_model = RecommendationModel(book_data)
    recommendation_model.train_model()

    # Run the user interface
    ui = UserInterface(recommendation_model, user_data)
    ui.run()

if __name__ == "__main__":
    main()