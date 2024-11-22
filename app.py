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
st.set_page_config(page_title="🎬 Movie Recommendation", page_icon="🎥", layout="wide")
st.title("✨ Welcome to Your Personal Movie Recommender 🎬")
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    font-family: Arial, sans-serif;
}
div[data-testid="stSidebar"] {
    background-color: #eef1f6;
}
h1 {
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for additional customization
st.sidebar.title("🔧 Settings")
st.sidebar.info("Use the sliders and options below to customize your recommendations.")

# Dropdown with searchable movie titles
st.subheader("🔎 Search for Your Favorite Movie")
movie_title = st.selectbox(
    "Start typing to search and select a movie:",
    options=movies["title"].tolist(),
    format_func=lambda x: x,  # Display full titles
    help="Type a few characters to filter the movie list."
)

# Slider for number of recommendations
num_recommendations = st.slider(
    "🎯 How many recommendations do you want?",
    min_value=1,
    max_value=20,
    value=5,
    help="Adjust the slider to set the number of movie recommendations."
)

# Call-to-action button
st.markdown(
    "<hr style='border-top: 3px solid #bbb;'/>",
    unsafe_allow_html=True,
)
if st.button("🚀 Get Recommendations"):
    recommendations, error = recommend_movies(movie_title, num_recommendations)
    if error:
        st.error(f"❌ {error}")
    else:
        st.success(f"🎉 Top {num_recommendations} movies similar to **{movie_title}**:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"**{i}. {movie}** 🎥")

# Footer
st.markdown(
    """
    ---
    💡 Built with ❤️ using [Streamlit](https://streamlit.io)
    """
)
