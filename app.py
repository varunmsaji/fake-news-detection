import streamlit as st 
import joblib 
import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
model = joblib.load('fakenews.pkl')
vector = joblib.load('vector.pkl')

def preprocess(text):
    ps = PorterStemmer()

    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)

    text = vector.transform([text])
    return text



st.set_page_config("fake news prediction",layout="centered")
st.header("this is a fake news prediction application to classsify the fake newsa and real news")

input = st.text_input("enter the article")
button = st.button('submit')
if input and button:
    input = preprocess(input)
    prediction = model.predict(input)
    if prediction ==1:
        st.write("the news is fake")

    else:
        st.write("the news is real")    


