# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import streamlit as st
pd.set_option('display.max_columns',None)
st.title("Movie Sentiment Analysis")

#form = st.form(key='my-form')
sent1 = st.text_input('Enter Movie Name')
sent2 = st.text_area('Enter Movie Review',height=50)
submit = st.button('Generate Review Sentiment')

data1 = pd.read_csv("IMDB Dataset.csv")

data=data1[:10000]

#len(data)

#data.head()

#"""#Data Preprocessing"""

TAG_RE = re.compile(r'<[^>]+>')#Regex to remove HTML Tags

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"s+[a-zA-Z]s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r's+', ' ', sentence)

    return sentence

#len(range(len(data['review'])))

for i in range(len(data['review'])):
  data['review'][i]=preprocess_text(data['review'][i])

#data['review'][0]

x = data['review']
y = data['sentiment']

# Positive Review =1 Negative Review =0
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

#"""#Splitting into train and test sets"""

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#"""#Vectorizing text reviews to numbers."""

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
x_train = vec.fit_transform(x_train).toarray()

x_test = vec.transform(x_test).toarray()

#"""#Fitting the Multinomial Naive Bayes Model"""

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)

#"""#Model Accuracy"""

#model.score(x_test, y_test)

#"""#Model Predictions"""

#model.predict(vec.transform(['Love this app simply awesome!']))

#"""#Functionizing and customizing Model Prediction for User friendliness"""

def pred(sent):
    v=model.predict(vec.transform([sent]))
    if v[0] ==1:
        st.success(f"Movie : {sent1} \n ")
        st.success(f"Review: \n {sent} \n ")
        st.success(f"Sentiment : Positive Review")
    else:
        st.error(f"Movie : {sent1} \n ")
        st.error(f"Review: \n {sent} \n ")
        st.error(f"Sentiment : Negative Review")

#"""#Performing Custom Predictions"""

#s = 'I would consider it an average movie. '#@param {type:"string"}
#pred(s)

#s = 'Very Good movie. Brilliant story'#@param {type:"string"}
#pred(s)

#s = 'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie'#@param {type:"string"}
#pred(s)

if submit and sent2:
    with st.spinner("Analyzing the review...."):
        pred(sent2)
        st.balloons()
else:
 st.warning("Not sure! Try to add some more words")
        
        