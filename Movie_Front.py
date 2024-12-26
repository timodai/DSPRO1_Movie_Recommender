import streamlit as st
import numpy as np
import pandas as pd
from streamlit_tags import st_tags
import joblib
import os

# ----------------------------------------------------
# Global Constants
# ----------------------------------------------------
GENRE_LIST = [
    'Film-Noir', 'War', 'Crime', 'Documentary', 'Drama', 'Mystery', 'Animation',
    'Western', 'Musical', 'Romance', 'Thriller', 'Adventure', 'Fantasy', 
    'Sci-Fi', 'Action', 'Children', 'Comedy', '(no genres listed)', 'Horror'
]

TRAINING_FEATURES = [
    'movieId', 'release_year', 'rating_count',
    'Film-Noir', 'War', 'Crime', 'Documentary', 'Drama', 'Mystery', 'Animation',
    'Western', 'Musical', 'Romance', 'Thriller', 'Adventure', 'Fantasy', 
    'Sci-Fi', 'Action', 'Children', 'Comedy', '(no genres listed)', 'Horror'
]

# ----------------------------------------------------
# Data Preparation Helpers
# ----------------------------------------------------
def load_data():
    """
    Load and preprocess the movies dataset. If a preprocessed file exists, use it; 
    otherwise, process the raw data and save it for future use.
    """
    if os.path.exists('movies_preprocessed.csv'):
        return pd.read_csv('movies_preprocessed.csv')

    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    # Add genre columns
    for genre in GENRE_LIST:
        movies[genre] = movies['genres'].str.contains(genre, na=False).astype(int)

    # Extract release year from movie titles
    movies['release_year'] = pd.to_numeric(
        movies['title'].str.extract(r'\((\d{4})\)')[0], errors='coerce'
    )

    # Count how many ratings each movie has received
    rating_counts = ratings.groupby('movieId').size().reset_index(name='rating_count')
    movies = movies.merge(rating_counts, on='movieId', how='left')

    # Handle missing values
    movies['release_year'].fillna(movies['release_year'].median(), inplace=True)
    movies['rating_count'].fillna(0, inplace=True)

    # Save for future use
    movies.to_csv('movies_preprocessed.csv', index=False)
    return movies

def load_model():
    """Load the pre-trained decision tree model."""
    return joblib.load('decision_tree_recommender.pkl')

# ----------------------------------------------------
# Recommendation Logic
# ----------------------------------------------------
def recommend_movies(seen_movies, preferred_genres, rating_preference, recency_preference, rating_count_preference, n_recommendations=5):
    """
    Generate movie recommendations based on user inputs.
    """
    model = load_model()
    movies = load_data()

    # Filter out movies the user has already seen
    all_movie_ids = set(movies['movieId'].unique())
    unseen_movie_ids = list(all_movie_ids - set(seen_movies))
    predict_data = pd.DataFrame({'movieId': unseen_movie_ids})

    # Merge with the main dataset to include all required columns
    predict_data = predict_data.merge(movies, on='movieId', how='left')

    # Normalize numeric features for prediction
    predict_data['release_year'] = (predict_data['release_year'] - movies['release_year'].min()) / (movies['release_year'].max() - movies['release_year'].min())
    predict_data['rating_count'] = (predict_data['rating_count'] - movies['rating_count'].min()) / (movies['rating_count'].max() - movies['rating_count'].min())

    # Ensure all training features exist
    for col in TRAINING_FEATURES:
        if col not in predict_data.columns:
            predict_data[col] = 0

    # Predict ratings using the model
    X = predict_data[TRAINING_FEATURES]
    predict_data['predicted_rating'] = model.predict(X)

    # Filter by preferred genres if specified
    if preferred_genres:
        genre_filter = predict_data[[g for g in preferred_genres if g in GENRE_LIST]].sum(axis=1) > 0
        predict_data = predict_data[genre_filter]

    # Sort based on user preferences
    sort_order = []
    if rating_preference == 'Good Rated':
        sort_order.append(('predicted_rating', False))
    elif rating_preference == 'Bad Rated':
        sort_order.append(('predicted_rating', True))

    if recency_preference == 'Recent':
        sort_order.append(('release_year', False))
    elif recency_preference == 'Old':
        sort_order.append(('release_year', True))

    if rating_count_preference == 'A lot of ratings':
        sort_order.append(('rating_count', False))
    elif rating_count_preference == 'Few ratings':
        sort_order.append(('rating_count', True))

    if sort_order:
        cols, ascending_vals = zip(*sort_order)
        predict_data = predict_data.sort_values(by=list(cols), ascending=list(ascending_vals))

    # Select top recommendations
    recommendations = predict_data.head(n_recommendations)

    # Keep only relevant details
    return recommendations[['movieId', 'title', 'genres', 'release_year', 'rating_count']]

# ----------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------
def page_config():
    """Configure the Streamlit page layout."""
    st.set_page_config(page_title="Movie Recommender")
    st.markdown("## Welcome to the Movie Recommender!")
    st.markdown("Input your preferences to discover your next favorite movie.")

def input_with_tags():
    """Collect user inputs via Streamlit widgets."""
    movies = load_data()

    genres = st.multiselect("Preferred Genres:", GENRE_LIST)

    movie_selection = st_tags(
        suggestions=movies['title'].tolist(),
        maxtags=10,
        label="Movies:",
        text="Enter movies you've seen:"
    )
    seen_movies = movies[movies['title'].isin(movie_selection)]['movieId'].tolist()

    rating_preference = st.radio("Rating Preference:", ['Default', 'Good Rated', 'Bad Rated'])
    recency_preference = st.radio("Recency Preference:", ['Default', 'Recent', 'Old'])
    rating_count_preference = st.radio("Rating Count Preference:", ['Default', 'A lot of ratings', 'Few ratings'])

    return genres, seen_movies, rating_preference, recency_preference, rating_count_preference

def generate_recommendations():
    """Generate and display movie recommendations based on user inputs."""
    genres, seen_movies, rating_pref, recency_pref, count_pref = input_with_tags()
    n_recommendations = st.slider("Number of Recommendations:", 1, 10, 5)

    if st.button("Get Recommendations"):
        recommendations = recommend_movies(
            seen_movies,
            genres,
            rating_pref if rating_pref != 'Default' else None,
            recency_pref if recency_pref != 'Default' else None,
            count_pref if count_pref != 'Default' else None,
            n_recommendations
        )
        st.write("### Here are your recommendations:")
        st.table(recommendations)

# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    page_config()
    generate_recommendations()
