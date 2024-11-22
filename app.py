import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDb API key (replace with your actual API key)
api_key = "your_tmdb_api_key"

# Function to fetch poster URL using TMDb API
def fetch_poster(movie_title):
    # Format the movie title for the API request
    formatted_title = "+".join(movie_title.split())

    # Make a request to TMDb search API to get the movie details by title
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={formatted_title}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')  # Get the poster path of the first result
            if poster_path:
                return "https://image.tmdb.org/t/p/w500/" + poster_path  # Full poster URL
    return None  # Return None if no poster found

# Load the dataset (your original dataset)
movies = pd.read_csv("data/movies.csv")

# Preprocessing: Combine all genres into a single string for each movie
movies["combined_features"] = movies["genres"].fillna("")

# Compute the cosine similarity matrix
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
feature_matrix = vectorizer.fit_transform(movies["combined_features"])
cosine_sim = cosine_similarity(feature_matrix)

# Function to recommend movies
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movies["title"].values:
        return None, f"Movie '{movie_title}' not found in the dataset."

    # Get the index of the movie
    idx = movies[movies["title"] == movie_title].index[0]

    # Get similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top recommendations (excluding the movie itself)
    sim_scores = sim_scores[1:num_recommendations + 1]
    recommended_indices = [score[0] for score in sim_scores]
    recommended_movies = movies.iloc[recommended_indices]

    return recommended_movies, None

# Streamlit App
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Find movies similar to your favorites!")

# Input: Search for a movie title (User can type a keyword)
search_keyword = st.text_input("Enter a movie keyword to search:", "")

# Filter movies based on the search keyword
if search_keyword:
    # Filter movies that contain the keyword in their title (case insensitive)
    filtered_movies = movies[movies["title"].str.contains(search_keyword, case=False, na=False)]

    # If no movies match, show an error message
    if filtered_movies.empty:
        st.error(f"No movies found for '{search_keyword}'. Try different keywords.")
    else:
        # Genre Filter
        genres = set([genre for sublist in filtered_movies["genres"].str.split("|") for genre in sublist])
        genres = sorted(genres)
        selected_genre = st.selectbox("Filter by Genre:", ["All"] + genres)

        # Apply genre filter if selected
        if selected_genre != "All":
            filtered_movies = filtered_movies[filtered_movies["genres"].str.contains(selected_genre, case=False)]

        # Create a dropdown menu with movie options
        movie_title = st.selectbox("Select a movie:", filtered_movies["title"])

        # Slider for the number of recommendations
        num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

        if st.button("Recommend"):
            recommendations, error = recommend_movies(movie_title, num_recommendations)
            if error:
                st.error(error)
            else:
                st.write(f"Top {num_recommendations} movies similar to **{movie_title}**:")

                # Display recommendations with posters in a grid layout
                num_cols = min(num_recommendations, 5)  # Limit number of columns to 5 for grid layout
                cols = st.columns(num_cols)  # Create columns to display posters horizontally
                for i, movie in enumerate(recommendations.itertuples()):
                    with cols[i % num_cols]:  # Distribute movies across columns
                        poster_url = fetch_poster(movie.title)  # Fetch poster URL for the movie
                        if poster_url:
                            st.image(poster_url, use_column_width=True)  # Display movie poster
                        st.markdown(f"**{movie.title}**")  # Display movie title
                        st.write(f"Genres: {movie.genres}")  # Display genres
else:
    st.write("Start by typing a movie title or a keyword to search for movies.")
