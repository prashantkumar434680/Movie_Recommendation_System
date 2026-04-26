import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
MOVIES_CSV = BASE_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = BASE_DIR / "tmdb_5000_credits.csv"


st.set_page_config(page_title="Movie Recommender System", layout="wide")


def parse_names(value, limit=None, compact=True):
    """Read TMDB JSON-like list columns and return compact name tokens."""
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
    except (ValueError, SyntaxError):
        return []

    names = [item.get("name", "") for item in items if item.get("name")]
    if compact:
        names = [name.replace(" ", "") for name in names]
    return names[:limit] if limit else names


def parse_director(value, compact=True):
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
    except (ValueError, SyntaxError):
        return []

    names = [
        item.get("name", "")
        for item in items
        if item.get("job") == "Director" and item.get("name")
    ]
    if compact:
        names = [name.replace(" ", "") for name in names]
    return names


@st.cache_data(show_spinner="Preparing movie data...")
def load_movies():
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        suffixes=("", "_credits"),
    )

    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "vote_count",
            "release_date",
        ]
    ].dropna(subset=["title", "overview"])

    movies["genre_names"] = movies["genres"].apply(parse_names)
    movies["keyword_names"] = movies["keywords"].apply(parse_names)
    movies["cast_names"] = movies["cast"].apply(lambda value: parse_names(value, limit=4))
    movies["director_names"] = movies["crew"].apply(parse_director)
    movies["display_genres"] = movies["genres"].apply(lambda value: parse_names(value, compact=False))
    movies["display_cast"] = movies["cast"].apply(
        lambda value: parse_names(value, limit=4, compact=False)
    )
    movies["display_directors"] = movies["crew"].apply(
        lambda value: parse_director(value, compact=False)
    )
    movies["overview_tokens"] = movies["overview"].apply(lambda text: str(text).split())

    movies["tags"] = (
        movies["overview_tokens"]
        + movies["genre_names"]
        + movies["keyword_names"]
        + movies["cast_names"]
        + movies["director_names"]
    ).apply(lambda tokens: " ".join(tokens).lower())

    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    return movies.reset_index(drop=True)


@st.cache_resource(show_spinner="Building recommendation model...")
def build_model(tags):
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(tags)
    return vectors


def recommend(movie_title, movies, vectors, count=5):
    matches = movies.index[movies["title"] == movie_title].tolist()
    if not matches:
        return pd.DataFrame()

    movie_index = matches[0]
    scores = cosine_similarity(vectors[movie_index], vectors).flatten()
    similar_indices = scores.argsort()[::-1][1 : count + 1]

    recommendations = movies.iloc[similar_indices].copy()
    recommendations["similarity"] = scores[similar_indices]
    return recommendations


def format_people(names):
    return ", ".join(names) if names else "Unknown"


def movie_card(movie):
    year = int(movie["year"]) if pd.notna(movie["year"]) else "N/A"
    genres = ", ".join(movie["display_genres"][:3]) or "Unknown"
    directors = format_people(movie["display_directors"])
    cast = format_people(movie["display_cast"])

    with st.container(border=True):
        st.subheader(movie["title"])
        st.caption(f"{year} | Rating {movie['vote_average']:.1f} | Match {movie['similarity']:.0%}")
        st.write(movie["overview"])
        st.markdown(f"**Genres:** {genres}")
        st.markdown(f"**Director:** {directors}")
        st.markdown(f"**Cast:** {cast}")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Movie Recommender System")

try:
    movies_df = load_movies()
    movie_vectors = build_model(movies_df["tags"])
except FileNotFoundError as exc:
    st.error(f"Missing dataset file: {exc.filename}")
    st.stop()

selected_movie = st.selectbox(
    "Choose a movie",
    movies_df["title"].sort_values().tolist(),
    index=movies_df["title"].sort_values().tolist().index("Avatar")
    if "Avatar" in movies_df["title"].values
    else 0,
)

selected = movies_df[movies_df["title"] == selected_movie].iloc[0]
left, right = st.columns([2, 1])
with left:
    st.write(selected["overview"])
with right:
    year = int(selected["year"]) if pd.notna(selected["year"]) else "N/A"
    st.metric("Release Year", year)
    st.metric("Average Rating", f"{selected['vote_average']:.1f}")
    st.metric("Votes", f"{int(selected['vote_count']):,}")

if st.button("Show Recommendations", type="primary"):
    recommended_movies = recommend(selected_movie, movies_df, movie_vectors)

    if recommended_movies.empty:
        st.warning("No recommendations found for this movie.")
    else:
        st.divider()
        st.header("Recommended Movies")
        for _, movie in recommended_movies.iterrows():
            movie_card(movie)
