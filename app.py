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
        .header {
            color: white;
            background-color: #1F2A44;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 30px;
            font-weight: bold;
        }
        .movie-card {
            background-color: #212121;
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .movie-card:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .movie-title {
            color: #FFD700;
            font-size: 18px;
            font-weight: bold;
        }
        .movie-rating {
            color: #FF6347;
            font-size: 16px;
        }
        .dropdown {
            background-color: #2e3a59;
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #00aaff;
            width: 100%;
            font-size: 16px;
            transition: box-shadow 0.3s ease, background-color 0.3s ease;
        }
        .dropdown:hover {
            background-color: #3a4a72;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .button {
            background-color: #1F2A44;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: transform 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .button:hover {
            background-color: #28527a;
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .button:focus {
            outline: none;
        }
        .movie-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .search-bar {
            background-color: #2e3a59;
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #00aaff;
            width: 100%;
            font-size: 16px;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .search-bar:focus {
            background-color: #3a4a72;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            outline: none;
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
            for rec in recommendations.itertuples():
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{rec.title}</div>
                    <div class="movie-rating">Rating: {rec.rating:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
