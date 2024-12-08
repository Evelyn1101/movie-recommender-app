import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from ast import literal_eval

# Title
st.title("Hybrid Movie Recommendation System")
st.write("Get personalized movie recommendations based on your favorite movie and user ID!")

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    links_small = pd.read_csv("links_small.csv")
    ratings = pd.read_csv("ratings_small.csv")

    # Process genres
    movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    movies = movies[movies['id'].str.isnumeric()]
    movies['id'] = movies['id'].astype(int)
    
    links_small = links_small[links_small['tmdbId'].notnull()]
    links_small['tmdbId'] = links_small['tmdbId'].astype(int)
    
    return movies, links_small, ratings

# Hybrid recommendation logic
def build_hybrid_model(movies, links_small, ratings):
    # Merge datasets
    smd = movies[movies['id'].isin(links_small['tmdbId'])]
    
    # Metadata-based features
    smd['soup'] = smd['genres'].apply(lambda x: ' '.join(x)) + ' ' + smd['overview'].fillna('')
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    
    # Collaborative filtering
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    
    id_map = links_small.merge(movies[['title', 'id']], left_on='tmdbId', right_on='id')[['movieId', 'title', 'id']]
    id_map = id_map.set_index('title')
    indices_map = id_map.set_index('id')
    
    return smd, cosine_sim, indices, svd, indices_map

@st.cache_data
def initialize_model():
    movies, links_small, ratings = load_data()
    return build_hybrid_model(movies, links_small, ratings)

# Recommendations function
def hybrid(userId, title, smd, cosine_sim, indices, svd, indices_map):
    if title not in indices:
        st.error("Movie not found. Please try another title.")
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

# Load model
smd, cosine_sim, indices, svd, indices_map = initialize_model()

# Input section
user_id = st.number_input("Enter your User ID:", min_value=1, step=1)
selected_movie = st.selectbox("Choose a movie you like:", smd['title'].unique())

# Recommendation results
if st.button("Get Recommendations"):
    recommendations = hybrid(user_id, selected_movie, smd, cosine_sim, indices, svd, indices_map)
    if not recommendations.empty:
        st.write("Recommended Movies:")
        for idx, row in recommendations.iterrows():
            st.subheader(row['title'])
            st.write(f"**Estimated Rating:** {row['est']:.2f}")
            st.write("---")
