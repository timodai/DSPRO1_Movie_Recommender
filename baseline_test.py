import pandas as pd
import random

def random_movie_recommender(file_path, num_recommendations=5):
    # Load the movies.csv file
    movies_df = pd.read_csv(file_path)
    
    # Ensure the file has a 'title' column for recommendations
    if 'title' not in movies_df.columns:
        raise ValueError("The input file must contain a 'title' column.")
    
    # Get a random sample of movies
    random_movies = movies_df.sample(n=num_recommendations)
    
    # Return the movie titles
    return random_movies['title'].tolist()

# Example usage
if __name__ == "__main__":
    # Replace 'movies.csv' with the path to your movies.csv file
    file_path = 'movies.csv'
    recommendations = random_movie_recommender(file_path)
    print("Recommended Movies:")
    for movie in recommendations:
        print(movie)
