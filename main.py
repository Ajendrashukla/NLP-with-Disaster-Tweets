import pickle
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import emoji
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words('english')) 
tk = TweetTokenizer() 
lemmatizer = WordNetLemmatizer()

# Load the tokenizer, lemmatizer, and stop words
tk = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the TF-IDF vectorizer and the model
Tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 1 Preprocessing the data
# 2 Vectorizing the data
# 3 Predicting the result by the model on the given data

st.title("Natural language processing with Disaster Tweets")
tweet = st.text_input("Enter the tweet")


# Preprocessing function
def transform_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove user mentions
    text = re.sub(r"@\S+", "", text)
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)
    # Remove emojis
    text = emoji.emojize(text, variant='emoji_type')
    # Converting the text to lowercase
    text = text.lower()
    # Tokenize the text
    words = tk.tokenize(text)
    # Lemmatize the text
    words = [lemmatizer.lemmatize(w) for w in words]
    # Remove stop words
    words = [w for w in words if w not in stop_words]
    # Join the tokens back together
    cleaned_text = ' '.join(words)

    return cleaned_text


if st.button('Predict'):
    preprocessed_data = transform_text(tweet)
    # 2 Vectorizing the preprocessed data
    transform_tweet = Tfidf.transform([preprocessed_data])
    # 3 Making prediction using the model
    decision = model.predict(transform_tweet)[0]

    if decision == 1:
        st.header("Disaster")
    else:
        st.header("Not Disaster")
