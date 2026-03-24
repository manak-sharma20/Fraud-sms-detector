import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set page styling
st.set_page_config(page_title="Fraud SMS Detector", page_icon="📱", layout="centered")

@st.cache_resource
def load_and_train_model():
    """Load dataset, train the model, and return vectorizer and model."""
    try:
        df = pd.read_csv("data/spam.csv", encoding="latin-1")
    except FileNotFoundError:
        st.error("Dataset 'data/spam.csv' not found. Please ensure it exists in the 'data' folder.")
        st.stop()
        
    # Preprocess
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    df['label'] = df['label'].map({
        'spam': 'scam',
        'ham': 'safe'
    })
    
    def clean_text_local(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        return text
        
    df['clean_message'] = df['message'].apply(clean_text_local)
    
    X = df['clean_message']
    y = df['label']
    
    # Vectorize and Train
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X_tfidf, y)
    
    return model, tfidf

def clean_input_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def main():
    st.title("📱 Fraud SMS Detector")
    st.markdown("Analyze an SMS or message to determine if it is a potential **SCAM** or **SAFE**.")
    
    # Load model 
    with st.spinner("Loading model..."):
        model, tfidf = load_and_train_model()

    # User Input
    user_message = st.text_area("Message Content", placeholder="Paste the message here...", height=150)
    
    if st.button("Analyze Message", type="primary", use_container_width=True):
        if not user_message.strip():
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing..."):
                cleaned_msg = clean_input_text(user_message)
                
                if cleaned_msg == "":
                    st.info("The message doesn't contain enough text to analyze.")
                else:
                    vectorized_msg = tfidf.transform([cleaned_msg])
                    prediction = model.predict(vectorized_msg)[0]
                    
                    st.markdown("---")
                    st.subheader("Result")
                    
                    if prediction == 'scam':
                        st.error("🚨 **SCAM ALERT!** 🚨\n\nThis message exhibits signs of being a scam. Do not click on any suspicious links or provide personal information.")
                    else:
                        st.success("✅ **SAFE**\n\nThis message appears to be safe.")

if __name__ == "__main__":
    main()
