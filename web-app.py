import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.write("""
# Song Popularity Prediction App
## This app can predict a song's popularity from 1956 - 2019!
This dataset contains audio statistics of the top 2000 tracks on Spotify.

This dataset contains audio statistics of the top 2000 tracks on Spotify. The data contains about 15 columns each describing the track and it's qualities.
Songs released from 1956 to 2019 are included from some notable and famous artists like Queen, The Beatles, Guns N' Roses, etc.
""")

link = '[Kaggle - Spotify - All Time Top 2000s Mega Dataset](https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset)'
st.markdown(link, unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')


# Get user inputs
def user_input_features():
    BPM = st.sidebar.slider('BPM', 37, 206, 100)
    Energy = st.sidebar.slider('Energy', 3, 100, 50)
    Danceability = st.sidebar.slider('Danceability', 10, 96, 50)
    Loudness = st.sidebar.slider('Loudness', -27, 0, -2)
    Liveness = st.sidebar.slider('Liveness', 2, 99, 50)
    Valence = st.sidebar.slider('Valence', 3, 99, 50)
    Length = st.sidebar.slider('Length (sec)', 93, 11, 1412)
    Acousticness = st.sidebar.slider('Acousticness', 0, 99, 50)
    Speechiness = st.sidebar.slider('Speechiness', 2, 55, 30)
    Year = st.sidebar.slider('Year', 1956, 2019, 1985)
    data = {'BPM': BPM,
            'Energy': Energy,
            'Danceability': Danceability,
            'Loudness': Loudness,
            'Liveness': Liveness,
            'Valence': Valence,
            'Length': Length,
            'Acousticness': Acousticness,
            'Speechiness': Speechiness,
            'Year': Year,
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Show user inputs
st.subheader('User Input parameters')
st.write(df)

# Create Plotly plot
columns = ['BPM', 'Energy', 'Loudness', 'Valence', 'Acousticness', 'Speechiness']
df_song_char = df.filter(items=columns)
y = df_song_char.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Audio Features from User Input')
st.plotly_chart(fig, use_container_width=True)

model_final_pipe = pickle.load(open('model_final_trained.pkl', 'rb'))

prediction = model_final_pipe.predict(df)

st.subheader('Predicted Song Popularity')
prediction = int(np.round(prediction, 0))
st.write(prediction)
