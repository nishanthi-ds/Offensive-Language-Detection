import joblib
import pandas as pd
import re

def prdict(text= '!! @abc hi hello how are you'):

    df = pd.DataFrame([text], columns=["text"])
    df['text'] = df['text'].apply(lambda x: re.sub(r'^!?[!]*\s*RT\s+@\w+:\s*','', x))
    # Clean the tweets
    def clean_tweet(text):
        text = re.sub(r'RT\s+@\w+:\s*', '', text)      # Remove "RT @username:"
        text = re.sub(r'@\w+', '', text)              # Remove mentions
        text = re.sub(r'[!?]{2,}', '', text)          # Remove repeated ! or ?
        text = re.sub(r'\s+', ' ', text).strip()      # Remove extra spaces
        return text

    df['text'] = df['text'].apply(clean_tweet)
    df.head()

    count_vectorizer = joblib.load('countvector.pkl')
    vector_inp = count_vectorizer.transform(df['text'])

    model = joblib.load('model.pkl')
    out = model.predict(vector_inp)

    return out
