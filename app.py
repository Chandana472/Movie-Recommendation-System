import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Load data and models
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

vote_info = pickle.load(open("vote_info.pkl", "rb"))
vote = pd.DataFrame(vote_info)

with open('csr_data_tf.pkl', 'rb') as file:
    csr_data = pickle.load(file)

model = pickle.load(open("model.pkl", "rb"))

def fetch_poster(movie_id):
    api_key = 'your_tmdb_api_key'
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

def recommend(movie_name):
    n_movies_to_recommend = 5
    idx = movies[movies['title'] == movie_name].index[0]
    distances, indices = model.kneighbors(csr_data[idx], n_neighbors=n_movies_to_recommend + 1)
    idx = list(indices.squeeze())
    df = np.take(movies, idx, axis=0)

    movies_list = list(df.title[1:])
    recommend_movies_names = []
    recommend_posters = []
    movie_ids = []
    for i in movies_list:
        temp_movie_id = (movies[movies.title == i].movie_id).values[0]
        movie_ids.append(temp_movie_id)
        recommend_posters.append(fetch_poster(temp_movie_id))
        recommend_movies_names.append(i)
    return recommend_movies_names, recommend_posters, movie_ids

# Streamlit app
st.title('MoviesWay - Movie Recommendation System')

selected_movie = st.selectbox('Select a Movie', movies['title'].values)

if st.button('Recommend'):
    st.text("Here are few recommendations:")
    names, posters, movie_ids = recommend(selected_movie)
    for name, poster in zip(names, posters):
        st.image(poster, caption=name, width=200)
