import streamlit as st
import numpy as np
import csv
#from streamlit_carousel import carousel
from streamlit_tags import st_tags

# setting the title and text on the page
def page_config() -> None: 
    st.set_page_config("Movie Recommender") # insert page name etc.
    st.markdown("## Movie Recommendations") # feel free to change text in the strings
    st.markdown("In the input below enter a list of movies you or your friends enjoy!")
    st.markdown("Our Algorithm will pick a few new movies that are similar to all of the selected ones.")

#Method 2 with st.tags
def input_with_tags() -> list:
    #read csv
    with open('movies.csv', newline='', encoding='utf-8') as csvfile:
        movies = csv.reader(csvfile, delimiter=',')
        movies_np = np.array(list(movies))

        #genre selection
        genre_suggest = ["Adventure","Comedy","Fantasy"] # insert all possible genres
        genre = st.selectbox("Preferred Genre:", genre_suggest)

        #movie selection
        movie_selection = st_tags(suggestions=list(movies_np[1:,1]), maxtags=10, label="Movies:", text="Enter your favorite movie/s here:")
        
        #rate the selected movies
        ratings = []
        st.markdown("Rate the selected movies from 0-5:")
        for movie in movie_selection:
            ratings.append(st.slider(label=movie, min_value=0.0, max_value=5.0, step=0.5))
            print(ratings)


        selection =  np.column_stack((movie_selection,ratings))
        return genre, selection


# function called with Generate Movie Recommendation button
def generate_suggestion(genre : str ,selection : list) -> None:
    with open("user_selection.txt", "w") as text:
        text.write(genre)
        text.write("\n")
        for movie in selection:
            text.write(movie[0])
            text.write(",")
            text.write(movie[1])
            text.write("\n")


if __name__ == "__main__":
    page_config()

    #user input section
    genre, selection = input_with_tags()
    print(selection)

    #Generate Button
    st.button(label="Generate Movie Recommendation", on_click=generate_suggestion(genre, selection))
        # carousel(items=test_items)



