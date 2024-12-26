import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
import joblib

# Load the MovieLens datasets
print("Loading datasets...")
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
print("Datasets loaded successfully.")

# Reduce the ratings dataset to 40% and only include relevant ratings
print("Reducing ratings dataset to 30%...")
ratings = ratings.sample(frac=0.30, random_state=42)  # Take a 40% sample of the ratings dataset
print("Filtering ratings >= 3.0...")
ratings = ratings[ratings['rating'] >= 3.0]  # Include only ratings >= 3.0
print(f"Reduced ratings dataset size: {ratings.shape[0]} rows.")

# Merge movies and ratings datasets
print("Merging movies and ratings datasets...")
merged_data = ratings.merge(movies, on='movieId')
print("Datasets merged successfully.")

# Use the predefined list of genres
print("Using predefined list of genres for encoding...")
genre_list = ['Film-Noir', 'War', 'Crime', 'Documentary', 'Drama', 'Mystery', 'Animation', 'Western', 'Musical', 'Romance', 'Thriller', 'Adventure', 'Fantasy', 'Sci-Fi', 'Action', 'Children', 'Comedy', '(no genres listed)', 'Horror']
for genre in genre_list:
    merged_data[genre] = merged_data['genres'].str.contains(genre, na=False).astype(int)
print(f"Genres encoded for: {genre_list}")

# Extract and normalize the release year
print("Extracting and normalizing release year...")
merged_data['release_year'] = merged_data['title'].str.extract(r'\((\d{4})\)').astype(float)
merged_data['release_year'].fillna(merged_data['release_year'].median(), inplace=True)  # Handle missing years
merged_data['release_year'] = (merged_data['release_year'] - merged_data['release_year'].min()) / (
    merged_data['release_year'].max() - merged_data['release_year'].min())
print("Release year extraction and normalization complete.")

# Calculate and normalize the number of ratings per movie
print("Calculating and normalizing the number of ratings per movie...")
ratings_count = ratings.groupby('movieId').size().reset_index(name='rating_count')
merged_data = merged_data.merge(ratings_count, on='movieId', how='left')
merged_data['rating_count'] = (merged_data['rating_count'] - merged_data['rating_count'].min()) / (
    merged_data['rating_count'].max() - merged_data['rating_count'].min())
print("Number of ratings per movie calculated and normalized.")

# Convert genres to binary
print("Ensuring genres are binary encoded...")
merged_data[genre_list] = merged_data[genre_list].applymap(lambda x: 1 if x > 0 else 0)
print("Genres are now binary.")

# Prepare the data for the model
print("Preparing features and target variables...")
features = ['movieId', 'release_year', 'rating_count'] + genre_list  # Include 'release_year' and 'rating_count' as features
X = merged_data[features]
y = merged_data['rating']
print("Feature preparation complete.")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Train a Decision Tree Regressor model
print("Training Decision Tree Regressor...")
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Save the trained model
print("Saving the trained model to file...")
joblib.dump(model, 'decision_tree_recommender.pkl')
print("Model saved successfully as 'decision_tree_recommender.pkl'.")

# Evaluate the model
print("Evaluating the model...")
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Function to recommend movies based on input movies and genres
def recommend_movies_based_on_input(seen_movies, preferred_genres=None, n_recommendations=5):
    print("Generating recommendations...")
    # Filter out movies the user has already seen
    all_movies = merged_data['movieId'].unique()
    movies_to_predict = list(set(all_movies) - set(seen_movies))

    # Create a DataFrame for prediction
    print("Preparing data for prediction...")
    predict_data = pd.DataFrame({'movieId': movies_to_predict})  # No userId
    predict_data = predict_data.merge(movies, on='movieId', how='left')

    # Add genre features
    for genre in genre_list:
        predict_data[genre] = predict_data['genres'].str.contains(genre, na=False).astype(int)

    # Extract and normalize release year for prediction data
    predict_data['release_year'] = predict_data['title'].str.extract(r'\((\d{4})\)').astype(float)
    predict_data['release_year'].fillna(merged_data['release_year'].median(), inplace=True)
    predict_data['release_year'] = (predict_data['release_year'] - merged_data['release_year'].min()) / (
        merged_data['release_year'].max() - merged_data['release_year'].min())

    # Calculate and normalize the number of ratings for prediction data
    predict_data = predict_data.merge(ratings_count, on='movieId', how='left')
    predict_data['rating_count'] = (predict_data['rating_count'] - merged_data['rating_count'].min()) / (
        merged_data['rating_count'].max() - merged_data['rating_count'].min())

    # Align columns with training data (fill missing columns with 0)
    for column in X.columns:
        if column not in predict_data.columns:
            predict_data[column] = 0

    predict_data[genre_list] = predict_data[genre_list].applymap(lambda x: 1 if x > 0 else 0)  # Ensure genres are binary
    predict_data = predict_data[X.columns]  # Ensure correct column order

    # Predict ratings
    print("Predicting ratings for movies...")
    predict_data['predicted_rating'] = model.predict(predict_data)

    # Filter by preferred genres if provided
    if preferred_genres:
        genre_columns = [genre for genre in preferred_genres if genre in predict_data.columns]
        if genre_columns:
            genre_filter = predict_data[genre_columns].sum(axis=1) > 0
            predict_data = predict_data[genre_filter]

    # Recommend top N movies
    print("Selecting top recommendations...")
    top_recommendations = predict_data.sort_values('predicted_rating', ascending=False).head(n_recommendations)
    return top_recommendations.merge(movies, on='movieId', how='left')[['movieId', 'title', 'predicted_rating', 'genres']]

# Example: Get recommendations based on input movies and genres
seen_movies = [1, 2, 3]  # Example movie IDs the user has seen
preferred_genres = ['Action', 'Comedy']  # Example preferred genres
recommendations = recommend_movies_based_on_input(seen_movies, preferred_genres, n_recommendations=5)
print("Recommendations based on your input:")
print(recommendations)
