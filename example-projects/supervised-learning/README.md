# Supervised Learning Project - Book Recommender

Detta är ett exempel på ett specifikationsdokument, som du börjar skriva på innan du börjar programmera.
När du bestämt dig för den grundläggande idén.

## Specifikation

Fantasy Book Recommendation System

Application Description:
A simple recommendation system for fantasy and sci-fi books based on user preferences and book characteristics.
Data:

Book features: author, subgenre (epic fantasy, urban fantasy, etc.), themes (magic, dragons, quests, etc.), length, publication year
User ratings and brief reviews

Data source:

Goodreads API for book information and user ratings
Amazon book listings (through web scraping)
Open library datasets

Model:
Alternative 1: Use a simple k-Nearest Neighbors (k-NN) algorithm. This model can find books similar to ones a user has enjoyed based on the features of the books.
Alternative 2: Collaborative Filtering

### Features

1. Data collection and preprocessing
2. Feature extraction and encoding
3. Model training (k-NN or Collaborative Filtering)
4. User interface for inputting preferences
5. Book recommendations based on user input

#### Limitations
* Time
* Keep number of genres low

### Requirements

#### Data requirements

1. Book features:
    * Title
    * Author
    * Subgenre (epic fantasy, urban fantasy, sci-fi, etc.)
    * Themes (magic, dragons, quests, space, aliens, etc.)
    * Length (page count)
    * Publication year
    * Average user rating
2. User data:
    * User ID
    * Books read (list of book IDs)
    * Ratings given

#### Software requirements

**LIBRARIES**

* pandas: Data manipulation and analysis
* scikit-learn: machine learning algorithms and preprocessing
* numpy: numerical operations
* requests: for API calls (if using Goodreads API)
* beatifulsoup4: web scraping (if scraping Amazon)

**CLASSES and METHODS**

1. `BookData`
    * Responsible for loading, cleaning and preprocessing book data
    * `load_data()`
    * `clean_data()`
    * `preprocess_data()`
2. `UserData`
    * Handles user information and ratings
    * `load_user_data()`
    * `update_user_preferences()`
3. `FeatureExtractor`
    * Extracts and encodes features from book and user data
    * `encode_categorical_features()`: One-hot encode categorical features
    * `normalize_numerical_features()`: Scale numerical features
    * `create_feature_matrix()`: Combine all features into a single matrix
4. `RecommendationModel`
    * Implements the chosen recommendation algorithms (k-NN)
    * `train_model()`: Train the recommendation model
    * `get_recommendations()`: Generate recommendations for a user
5. `UserInterface`
    * Manages user interactions and displays recommendations
    * `get_user_input()`: Prompt user for preferences
    * `display_recommendations()`: Show recommended books to the user

## Diskussion
