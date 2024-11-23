import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies and ratings dataset
movies = pd.read_csv("data/movies.csv")  # Movie information
ratings = pd.read_csv("data/ratings.csv")  # Ratings information

# Merge movies and ratings datasets on movieId
movies_with_ratings = pd.merge(movies, ratings.groupby("movieId")["rating"].mean().reset_index(), on="movieId")

# Preprocessing: Combine all genres into a single string for each movie
movies_with_ratings["combined_features"] = movies_with_ratings["genres"].fillna("")

# Compute the cosine similarity matrix based on genres
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
feature_matrix = vectorizer.fit_transform(movies_with_ratings["combined_features"])
cosine_sim = cosine_similarity(feature_matrix)

# Function to recommend movies based on a selected movie title
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movies_with_ratings["title"].values:
        return None, f"Movie '{movie_title}' not found in the dataset."

    # Get the index of the movie
    idx = movies_with_ratings[movies_with_ratings["title"] == movie_title].index[0]

    # Get similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top recommendations (excluding the movie itself)
    sim_scores = sim_scores[1:num_recommendations + 1]
    recommended_indices = [score[0] for score in sim_scores]
    recommended_movies = movies_with_ratings.iloc[recommended_indices]

    return recommended_movies, None

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        .movie-card {
            background-color: #1F2A44;
            color: white;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .movie-card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }
        .movie-title {
            font-size: 18px;
            font-weight: bold;
            color: #FFD700;
            margin-bottom: 10px;
        }
        .movie-rating {
            font-size: 16px;
            color: #FF6347;
            margin-bottom: 10px;
        }
        .recommend-button {
            background-color: #28527a;
            color: white;
            font-size: 14px;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .recommend-button:hover {
            background-color: #4682B4;
            transform: scale(1.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites!")

# Input: Search for a movie by keyword (movie title)
search_keyword = st.text_input(
    "Enter a movie keyword to search:",
    "",
    key="movie_search",
    placeholder="E.g., Toy Story",
    help="Type a movie keyword to filter options."
)

# Dropdown for genres
selected_genre = st.selectbox(
    "Filter by Genre (Optional):",
    ["All"] + sorted(set([genre for sublist in movies["genres"].str.split("|") for genre in sublist])),
    help="Filter movies by specific genre."
)

# Apply a filter
filtered_movies = movies_with_ratings[
    (movies_with_ratings["title"].str.contains(search_keyword, case=False, na=False)) & 
    (movies_with_ratings["genres"].str.contains(selected_genre, case=False, na=False) if selected_genre != "All" else True)
]

if filtered_movies.empty:
    st.error("No movies match your search criteria.")
else:
    movie_title = st.selectbox("Choose a movie:", filtered_movies["title"])
    num_recommendations = st.slider("How many recommendations?", 1, 10, 5)

    if st.button("Get Recommendations", help="Generate similar movies"):
        recommendations, error = recommend_movies(movie_title, num_recommendations)
        if error:
            st.error(error)
        else:
            st.write(f"### Recommendations for **{movie_title}**:")
            cols_per_row = 5  # Number of columns per row
            cols = st.columns(cols_per_row)  # Create columns
            for idx, rec in enumerate(recommendations.itertuples()):
                col = cols[idx % cols_per_row]
                with col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{rec.title}</div>
                        <div class="movie-rating">Rating: {rec.rating:.1f}</div>
                        <button class="recommend-button">Details</button>
                    </div>
                    """, unsafe_allow_html=True)
