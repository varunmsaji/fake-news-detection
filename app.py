import streamlit as st 
import joblib 
import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_extras.add_vertical_space import add_vertical_space


#loading the model and the vectorizer
model = joblib.load('fakenews.pkl')
vector = joblib.load('vector.pkl')

#function for text preprocessing
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
with st.sidebar:
    st.title('FAKE NEWS PREDICTION')
    st.markdown('''
    ## About
    This is an app to predict whether the given news in fake or not using:
                
    -streamlit
    ,logistic regression as the classification model
               
    
 
    ''')
    add_vertical_space(5)
    st.write('made by varun m s')
    st.write('varunmsaji01@gmail.com')

input = st.text_input("enter the article")
button = st.button('submit')
if input and button:
    input = preprocess(input)
    prediction = model.predict(input)
    if prediction ==1:
        st.write("the news is fake")

    else:
        st.write("the news is real")    


