import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
from sklearn.linear_model import LinearRegression

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
import pickle as pk

print("I am here")

import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

print("I am here")

def extract_pos(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_words = [word for word, pos in pos_tags if pos.startswith('VB') or pos.startswith('JJ') or pos.startswith('NN')]
    return ' '.join(pos_words)

def process_dataset(text):
    lower_cased = text.lower()
    tokens = nltk.word_tokenize(lower_cased)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    preprocessed = ' '.join(filtered_tokens)
    return preprocessed
    
def get_embeddings(text):
    embeds = embed([text])
    return np.array(embeds[0])

ncols = 100
#model = Sequential()
#model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(ncols,1)))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(64, activation='relu'))
#model.add(Dense(1))

#model.load_weights("USE.h5")

filename = 'gb.sav'
model = pk.load(open(filename, 'rb'))

pca = pk.load(open("pca.pkl",'rb'))


st.markdown(
    """
    <style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center;'>Automatic Grading System</h1>
    """,
    unsafe_allow_html=True
)


with st.form("input_form"):
    question = st.text_input("Enter your question:")
    answer = st.text_area("Enter your answer:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if question and answer:
        tagged = extract_pos(question + " " + answer)
        processed = process_dataset(tagged)
        embeddings = get_embeddings(processed)
        print(len(embeddings))
        transformed = pca.transform(pd.DataFrame(list(embeddings)).T)
#        result = nlp(question=question, context=answer)
#        score = result['score']
        score = model.predict(transformed)
        st.write(f"Score: {score[0]}")
    else:
        st.write("Please enter both the question and the answer.")
        
st.markdown(
    """
    <h3 style='text-align: center;'>Developed by:</h3>
    <p style='text-align: center;'>Harshitha Kanisettypalli</p>
    <p style='text-align: center;'>Kavya Duvvuri</p>
    <p style='text-align: center;'>Masabattula Teja Nikhil</p>
    """,
    unsafe_allow_html=True
)