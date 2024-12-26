# DSPRO1_Movie_Recommender

## Project title

Movie Recommender (tbd)

## Description

In today’s digital world, recommendation systems are crucial for suggesting relevant content to users, from movies to products. Our project aims to build a movie recommender system tailored for group settings. Specifically, when multiple people input their favourite movies, the system will suggest a new movie that everyone should enjoy.

By leveraging IMDb’s and Movielens dataset, which includes a wealth of information on movie ratings, genres, directors, and other metadata, we aim to build a robust system that takes into account the diverse tastes of users and provides accurate recommendations.

## Team

- **André Dollfus** - Scrum Master, Preprocessing
- **André Farkas** - Analytics, Quality Assurance
- **TImo Tran** - Data, Recommendation System

---

## Prerequisites

### Python Environment Setup
1. Ensure Python (preferably 3.8 or higher) is installed.
2. Use the provided `environment.yml` file to set up the environment by running:
   ```bash
   conda env create -f environment.yml
   conda activate movie_recommender
   ```

### Required Files
Ensure the following files are present in the working directory:
- `movies.csv`: Raw movie metadata (e.g., movie titles, genres, etc.).
- `ratings.csv`: Raw movie ratings data (e.g., userId, movieId, rating, timestamp).
- `decision_tree_recommender.pkl`: Pre-trained decision tree model for predicting movie ratings.
- `movies_preprocessed.csv`: This file will be automatically generated upon the first run if not already present.

---

## How to Run

1. Launch the Streamlit application by running the following command:
   ```bash
   streamlit run Movie_Front.py
   ```
2. Open the application in your browser. If not opened automatically, copy and paste the generated URL (e.g., `http://localhost:8501`) into your browser.

---

## How It Works

### User Interface
The application provides the following options:
1. **Input Preferences**:
   - **Preferred Genres**: Select your favorite genres from a dropdown list (e.g., Action, Drama, Comedy).
   - **Movies**: Type and select movies you've already seen from a suggestions list.
   - **Rating Preference**: Choose whether to prioritize movies with good, bad, or default ratings.
   - **Recency Preference**: Specify whether you prefer newer, older, or default release years.
   - **Rating Count Preference**: Decide if you prefer movies with many or few ratings.

2. **Set Recommendation Count**:
   - Use the slider to specify the number of recommendations you want (1 to 10).

3. **Get Recommendations**:
   - Click the **Get Recommendations** button to generate a table displaying recommended movies, including details like:
     - Movie Title
     - Genres
     - Release Year
     - Rating Count

---

## Application Workflow

### Data Preparation (`movies_preprocessed.csv`)
This file is a preprocessed version of the raw movie data, containing:
- **Genres**: Converted into binary columns (e.g., `Action`, `Drama`).
- **Release Year**: Extracted from movie titles.
- **Rating Count**: Number of ratings per movie.

If `movies_preprocessed.csv` is missing, it will be created during the first run by:
1. Combining data from `movies.csv` and `ratings.csv`.
2. Cleaning and enriching the dataset.
3. Saving the processed file for future use.

### Recommendation Model (`decision_tree_recommender.pkl`)
- A pre-trained **decision tree model** that predicts `predicted_rating` for unseen movies.
- Uses features from `movies_preprocessed.csv`, such as genres, release year, and rating count.
- Outputs ratings that are sorted and filtered based on user preferences.

### Recommendation Logic
- Filters out movies the user has already seen.
- Sorts recommendations based on:
  - Rating (Good Rated or Bad Rated).
  - Recency (Recent or Old).
  - Rating Count (Many or Few Ratings).
- Returns the top movies matching user preferences.

### Streamlit Interface (`Movie_Front.py`)
- **Frontend**: Widgets like dropdowns, sliders, and buttons collect user input.
- **Backend**: Executes the recommendation logic and displays results dynamically.
  - **Inputs**:
    - **Preferred Genres**: Selected using a multi-select widget.
    - **Seen Movies**: Entered using dynamic tag inputs.
    - **Preferences**: Options for rating, recency, and rating count.
    - **Number of Recommendations**: Selected using a slider.
  - **Outputs**:
    - Displays top recommendations in a table format with columns such as `movieId`, `title`, `genres`, `release_year`, and `rating_count`.

---

## File Descriptions

### `movies_preprocessed.csv`
- **Purpose**: Stores preprocessed movie data for faster access in future runs.
- **Contents**:
  - `movieId`: Unique movie identifier.
  - `title`: Movie name.
  - `release_year`: Year of release.
  - `rating_count`: Number of ratings received.
  - Binary columns for genres (e.g., `Action`, `Drama`).

### `decision_tree_recommender.pkl`
- **Purpose**: Pre-trained decision tree model to predict movie ratings.
- **Role**: Analyzes features like genres, release year, and rating count to score unseen movies.

### `dt_recommender.py`
- **Purpose**: Backend script for preparing data and training the model.
- **Features**:
  - **Data Preparation**:
    - Loads the `movies.csv` and `ratings.csv` datasets.
    - Reduces the ratings dataset to 30% of the original size and filters for ratings >= 3.0.
    - Merges movies and ratings datasets, encoding genres as binary features.
    - Extracts and normalizes the release year.
    - Calculates and normalizes the number of ratings per movie.
  - **Model Training**:
    - Prepares features like genres, release year, and rating count.
    - Splits data into training and testing sets.
    - Trains a **Decision Tree Regressor** to predict movie ratings.
    - Saves the trained model as `decision_tree_recommender.pkl`.
  - **Recommendation Logic**:
    - Filters out movies the user has already seen.
    - Sorts recommendations by predicted rating, optionally filtered by preferred genres.
    - Recommends the top movies based on the user's preferences.
  - **Evaluation**:
    - Evaluates the model using Mean Squared Error (MSE) to measure prediction accuracy.

### `Movie_Front.py`
- **Purpose**: Frontend script for the Streamlit application.
- **Features**:
  - Loads the preprocessed data (`movies_preprocessed.csv`) and the trained model (`decision_tree_recommender.pkl`).
  - Provides a user-friendly interface for entering preferences and displaying recommendations.
  - Implements the recommendation logic dynamically based on user inputs.

---

## Summary

The Movie Recommender application combines data preprocessing, a trained decision tree model, and a Streamlit interface to provide personalized movie suggestions. Users can input preferences like genres, rating preferences, and recency to receive tailored recommendations in a dynamic, interactive format.

For additional support or customization, feel free to reach out!

