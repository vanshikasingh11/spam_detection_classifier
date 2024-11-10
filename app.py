import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if not already done
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)  # Join words back into a single string

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model and vectorizer
tfidf, model = load_model()
if model is None:
    st.stop()  # Stop execution if model loading failed

# Apply CSS for custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea, .stButton {
        font-size: 16px;
        color: #333;
    }
    .stTextArea {
        height: 150px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .predict-button {
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📧 Email/SMS Spam Classifier")

# User input area for the message
input_sms = st.text_area("Share your message for prediction", placeholder="Input your text for spam detection...")

# Aligning the button to center using Streamlit columns
col1, col2, col3 = st.columns([1, 1, 1])

# Center-align the button using the second column
with col2:
    if st.button('Predict'):
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict using the model
        result = model.predict(vector_input)[0]
        # 4. Display the result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
