import streamlit as st
import pickle
import string #
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the necessary NLTK data files are available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Setting up the title and description with animation
st.markdown("""
    <style>
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .animated-gradient {
        background: linear-gradient(270deg, #BB86FC, #3700B3);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-family: 'Arial Black', sans-serif;
    }
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .result {
        font-size: 24px;
        animation: fadeIn 2s;
        margin-top: 20px;
        padding: 10px;
        border-radius: 10px;
    }
    .spam {
        color: #FF5252; /* Red */
    }
    .ham {
        color: #4CAF50; /* Green */
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="animated-gradient">Email/SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown("**A simple web app to classify emails or SMS messages as spam or not spam.**")

# Adding a sidebar
st.sidebar.title("About")
st.sidebar.info("This application is a simple implementation of a spam classifier using machine learning. Enter your email or SMS message and click on 'Predict' to see if it's classified as Spam or Not Spam.")

# Input area for the user to type the message
input_sms = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict using the loaded model
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0][result]
        # 4. Display the result with animation
        if result == 1:
            st.markdown(f'<div class="result spam">🚫 This message is classified as Spam with a probability of {probability * 100:.2f}%.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result ham">✅ This message is classified as Not Spam with a probability of {probability * 100:.2f}%.</div>', unsafe_allow_html=True)

# Adding some custom styling
st.markdown("""
    <style>
    .reportview-container {
        background: #1b1b2f;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #0a0a0a;
        color: white;
    }
    .stButton>button {
        background-color: #0a0a0a;
        color: #BB86FC;
        font-size: 20px;
        border-radius: 10px;
        border: 2px solid #BB86FC;
    }
    .stTextArea>textarea {
        background-color: #0a0a0a;
        color: white;
        font-size: 16px;
        border: 2px solid #BB86FC;
        border-radius: 10px;
    }
    .stTextArea>label {
        color: #BB86FC;
    }
    footer {
        visibility: hidden;
    }
    footer:after {
        content: 'Made with ❤️ by YourName';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
        font-size: 16px;
        color: #BB86FC;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
