import requests
from bs4 import BeautifulSoup
import pandas as pd

# Load MovieLens dataset
movies = pd.read_csv("data/movies.csv")  # Ensure this file exists in your 'data' folder

def fetch_poster_tmdb(title):
    # Search TMDb by title
    search_url = f"https://www.themoviedb.org/search?query={'+'.join(title.split())}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the first movie result
    movie = soup.find("div", class_="card style_1")
    if movie:
        poster = movie.find("img")
        if poster and poster["data-src"]:
            return "https://www.themoviedb.org" + poster["data-src"]
    return None  # If no poster found

# Fetch posters for all movies
print("Fetching posters...")
movies["poster_url"] = movies["title"].apply(fetch_poster_tmdb)

# Save the updated dataset
movies.to_csv("data/movies_with_posters.csv", index=False)
print("Updated dataset saved to 'data/movies_with_posters.csv'")

