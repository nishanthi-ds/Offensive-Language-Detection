# Offensive Language Detection

This project aims to develop a machine learning model to detect offensive language in tweets using a Twitter dataset. The process involves the following steps:

1- **Data Collection**: Tweets are collected from a labeled Twitter dataset specifically designed for offensive language detection tasks.

2- **Data Cleaning and Balancing**: The tweets are preprocessed to remove noise such as URLs, mentions, and special characters. The dataset is then resampled using undersampling techniques to address class imbalance.

3- **Data Preprocessing**: Text data is converted into numerical format using vectorization techniques such as TF-IDF or CountVectorizer, making it suitable for machine learning models.

4- **Model Building and Training**: A Logistic Regression model is trained on the preprocessed data to classify tweets as offensive or non-offensive.

5- **Model Evaluation**: The model is evaluated using precision, recall, F1-score, and a confusion matrix to ensure its reliability in detecting offensive content.

6- **Deployment**: A Streamlit application is created to allow users to enter tweet text and receive real-time feedback on whether the content is offensive. The app is run locally using:
