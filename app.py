import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv("data/movies.csv")

# TMDb API setup
TMDB_API_KEY = "your_tmdb_api_key"  # Replace with your API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Preprocessing: Combine all genres into a single string for each movie
movies["combined_features"] = movies["genres"].fillna("")

# Compute the cosine similarity matrix
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
feature_matrix = vectorizer.fit_transform(movies["combined_features"])
cosine_sim = cosine_similarity(feature_matrix)

# Function to fetch movie poster
def fetch_movie_poster(movie_title):
    """Fetch the poster URL for a movie using the TMDb API."""
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    return None

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
    recommended_movies = movies.iloc[recommended_indices]["title"].tolist()

    return recommended_movies, None

# Streamlit App
st.set_page_config(page_title="🎬 Movie Recommendation", page_icon="🎥", layout="wide")
st.title("✨ Welcome to Your Personal Movie Recommender 🎬")

# Dropdown with searchable movie titles
st.subheader("🔎 Search for Your Favorite Movie")
movie_title = st.selectbox(
    "Start typing to search and select a movie:",
    options=movies["title"].tolist(),
)

# Slider for number of recommendations
num_recommendations = st.slider("🎯 How many recommendations do you want?", min_value=1, max_value=20, value=5)

# Recommend button logic
if st.button("🚀 Get Recommendations"):
    recommendations, error = recommend_movies(movie_title, num_recommendations)
    if error:
        st.error(f"❌ {error}")
    else:
        st.success(f"🎉 Top {num_recommendations} movies similar to **{movie_title}**:")
        for i, movie in enumerate(recommendations, start=1):
            poster_url = fetch_movie_poster(movie)
            if poster_url:
                st.image(poster_url, width=150, caption=f"{i}. {movie}")
            else:
                st.write(f"**{i}. {movie}** 🎥 (Poster not available)")
