import streamlit as st
import pandas as pd
import pickle,re
import numpy as np

# -------- Prediction Function --------
def predict(text):
    df = pd.DataFrame([text], columns=["text"])

    # Clean the tweet
    def clean_tweet(text):
        text = re.sub(r'RT\s+@\w+:\s*', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[!?]{2,}', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['text'] = df['text'].apply(clean_tweet)

    # Load vectorizer and transform text
    count_vectorizer = pickle.load(open('countvector.sav', 'rb'))  # Must be CountVectorizer object
    vector_inp = count_vectorizer.transform(df['text'])

    # Load model and predict
    model = pickle.load(open('model.sav', 'rb'))
    prediction = model.predict(vector_inp)

    return prediction[0]

# -------- UI Design --------
st.set_page_config(page_title="Stress Checker", layout="centered")


rainbow_style = """
<style>
    .stApp {
        background: linear-gradient(135deg, red, orange, yellow, green);
        background-size: 400% 400%;
        animation: rainbow 15s ease infinite;
        color: white;
    }

    @keyframes rainbow {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }

    html, body, [class*="css"]  {
        color: white;
        font-weight: bold;
    }

    .stButton > button {
        background-color: #000000AA;
        color: black;
        font-weight: bold;
        border-radius: 8px;
    }

    .stSelectbox label {
        color: white !important;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: black !important;
    }
</style>
"""
st.markdown(rainbow_style, unsafe_allow_html=True)

# -------- Streamlit Form --------
st.title("🚨 Tweet Offensive Language Detection")

with st.form("tweet_form"):
    user_input = st.text_area("Paste a tweet or write your own:")
    submit = st.form_submit_button("Submit")

    if submit:
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = predict(user_input)
            st.success(f"Prediction: {'Offensive' if result == 1 else 'Not Offensive'}")
