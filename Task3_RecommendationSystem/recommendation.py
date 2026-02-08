# Recommendation System using Content-Based Filtering
# Task 3 - CODSOFT Artificial Intelligence Internship

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'Movie': [
        'Avengers', 'Iron Man', 'Thor',
        'Titanic', 'Notebook',
        'Inception', 'Interstellar',
        'Hangover', 'Superbad'
    ],
    'Genre': [
        'Action', 'Action', 'Action',
        'Romance', 'Romance',
        'Sci-Fi', 'Sci-Fi',
        'Comedy', 'Comedy'
    ]
}

df = pd.DataFrame(data)

# One-hot encoding genres
genre_encoded = pd.get_dummies(df['Genre'])

# Calculate similarity
similarity = cosine_similarity(genre_encoded)

def recommend(movie_name):
    if movie_name not in df['Movie'].values:
        print("Movie not found!")
        return

    idx = df[df['Movie'] == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nRecommended movies similar to '{movie_name}':")
    for i in scores[1:4]:
        print(df.iloc[i[0]]['Movie'])

# User input
movie = input("Enter a movie name: ")
recommend(movie)
