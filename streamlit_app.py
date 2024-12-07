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

# Custom CSS for UI
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
        .movie-image {
            width: 100%;
            height: auto;
            border-radius: 10px;
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
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title("🎬Movie Recommendation System")
st.write("Find movies similar to your favorites!")

# Input: Search for a movie by keyword (movie title)
search_keyword = st.text_input("Enter a movie keyword to search:", "", key="movie_search", placeholder="E.g., Toy Story")

# Option 2: Select movies by genre
genres = set([genre for sublist in movies["genres"].str.split("|") for genre in sublist])
genres = sorted(genres)
selected_genre = st.selectbox("Filter by Genre (Optional):", ["All"] + genres)

# Filter movies based on the keyword or genre
if search_keyword:
    # Filter movies that contain the keyword in their title (case insensitive)
    filtered_movies = movies_with_ratings[movies_with_ratings["title"].str.contains(search_keyword, case=False, na=False)]
elif selected_genre != "All":
    # Filter movies based on the selected genre
    filtered_movies = movies_with_ratings[movies_with_ratings["genres"].str.contains(selected_genre, case=False)]
else:
    # Show all movies if no filter is applied
    filtered_movies = movies_with_ratings

# If no movies match, show an error message
if filtered_movies.empty:
    st.error(f"No movies found. Try using a different keyword or genre.")
else:
    # Create a dropdown menu with filtered movie options
    movie_title = st.selectbox("Select a movie:", filtered_movies["title"])

    # Slider for the number of recommendations
    num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5, key="num_recommendations")

    # Button for recommending movies
    recommend_button = st.button("Get Recommendations", key="recommend_button", help="Click to get similar movie recommendations!")

    if recommend_button:
        recommendations, error = recommend_movies(movie_title, num_recommendations)
        if error:
            st.error(error)
        else:
            st.write(f"### Top {num_recommendations} movies similar to **{movie_title}**:")

            # Display recommendations in a grid layout
            cols = st.columns(5)
            movie_count = 0
            for i, movie in enumerate(recommendations.itertuples(), start=1):
                col = cols[movie_count % 5]
                with col:
                    st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{movie.title}</div>
                            <div class="movie-rating">Rating: {movie.rating:.1f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                movie_count += 1 
