import streamlit as st
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


SPOTIFY_CLIENT_ID = '15ca39b81ba444d9a5d4ddc6c7a96d3e'
SPOTIFY_CLIENT_SECRET = 'f01738951fbe4304923161f469c48117'
SPOTIFY_REDIRECT_URI = 'http://localhost:8501/callback/'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-library-read"
))

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

mood_to_genre = {
    "anger": "metal",
    "disgust": "experimental",
    "fear": "dark ambient",
    "joy": "pop",
    "neutral": "chillhop",
    "sadness": "blues",
    "surprise": "electronic"
}

def detect_emotion(text):
    result = emotion_model(text)
    return max(result, key=lambda x: x['score'])['label']

def get_songs_by_genre(genre, limit=10):
    results = sp.search(q=f"genre:{genre}", type="track", limit=limit)
    songs = []
    for track in results['tracks']['items']:
        songs.append({
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "id": track['id']
        })
    return pd.DataFrame(songs)

def recommend_songs(content_df, input_song, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf.fit_transform(content_df['name'] + " " + content_df['artist'])
    cosine_sim = cosine_similarity(content_matrix, content_matrix)
    
    song_index = content_df[content_df['name'] == input_song].index[0]
    sim_scores = list(enumerate(cosine_sim[song_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_songs = sim_scores[1:top_n+1]
    
    recommendations = []
    for i in top_songs:
        recommendations.append(content_df.iloc[i[0]])
    return pd.DataFrame(recommendations)



st.title("🎵 Music Mood Recommender 🎶")
st.write("How do you feel? We ask because we care! Tell us and we'll recommend the perfect playlist just for you.")

user_input = st.text_input("Tell us your feelings.")
if st.button("Analyse"):
    if user_input:
        emotion = detect_emotion(user_input)
        st.write(f"Your detected mood is: **{emotion.capitalize()}**.")
        genre = mood_to_genre.get(emotion.lower(), "pop")
        st.write(f"So the genre we picked is: **{genre.capitalize()}**.")
        
        songs_df = get_songs_by_genre(genre)
        if not songs_df.empty:
            st.write("Your playlist:")
            input_song = songs_df.iloc[0]['name']
            recommendations = recommend_songs(songs_df, input_song)
            for _, song in recommendations.iterrows():
                st.write(f"- {song['name']} - {song['artist']}")
        else:
            st.write("There are no songs in this category.")
    else:
        st.write("Please type how you feel.")