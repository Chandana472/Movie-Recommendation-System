import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
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
    recommended_movies = movies.iloc[recommended_indices]["title"].tolist()

    return recommended_movies, None

# Streamlit App
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Find movies similar to your favorites!")

# Input: Movie title
movie_title = st.text_input("Enter a movie title:", "Toy Story (1995)")
num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if st.button("Recommend"):
    recommendations, error = recommend_movies(movie_title, num_recommendations)
    if error:
        st.error(error)
    else:
        st.write(f"Top {num_recommendations} movies similar to **{movie_title}**:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
