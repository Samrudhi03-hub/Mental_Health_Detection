import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import string

# Load the saved fitted model and vectorizer
model = joblib.load('mental_health_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Define a dictionary to provide descriptions for different conditions
condition_descriptions = {
    'Anxiety': "Anxiety involves excessive worry, nervousness, or fear. It's a common response to stress but can become overwhelming.",
    'Depression': "Depression is a mood disorder causing a persistent feeling of sadness and loss of interest. It affects how you feel, think, and handle daily activities.",
    'Addiction': "Addiction is the inability to stop engaging in a behavior or using a substance despite the negative consequences.",
    'Normal': "You seem to be in a good mental health state. Keep taking care of yourself and maintain a positive mindset.",
    'Suicidal': "Suicidal thoughts or tendencies can indicate severe emotional distress. Immediate intervention and support are crucial.",
    'Bipolar': "Bipolar disorder involves extreme mood swings between mania (highs) and depression (lows). Treatment helps manage these episodes.",
    'Stress': "Stress is the body's reaction to any change that requires adjustment or response. Chronic stress can affect both mental and physical health.",
    'Personality disorder': "Personality disorders involve unhealthy patterns of thinking, functioning, and behaving, which can lead to distress or impairment."
}

# Set the background color to lavender
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E6E6FA;  /* Lavender color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit application
st.title("Mental Health Detection System")
st.write("Enter a statement to evaluate mental health status:")

# Input text box for user statement
user_input = st.text_area("Your Statement:")

# When the user clicks the submit button
if st.button("Submit"):
    if user_input:
        # Preprocess the input statement
        cleaned_input = preprocess_text(user_input)
        
        # Transform the input using the loaded vectorizer
        input_tfidf = vectorizer.transform([cleaned_input])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_tfidf)[0]
        
        # Fetch the description based on the predicted condition
        description = condition_descriptions.get(prediction, "No description available for this condition.")
        
        # Display the prediction and description
        st.success(f"The predicted mental health status is: {prediction}")
        st.write(f"Description: {description}")
    else:
        st.error("Please enter a statement.")
