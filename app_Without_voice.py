import streamlit as st
from transformers import pipeline

# Supported languages with their language codes for M2M100 model
LANGUAGES = {
    'English': 'en', 
    'Hindi': 'hi', 
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Odia': 'or',
    'Malayalam': 'ml',
    'Punjabi': 'pa',
    # Add more languages as needed
}


st.title("Language Translator")

# Language selection
src_lang = st.selectbox("Select source language", options=list(LANGUAGES.keys()))
tgt_lang = st.selectbox("Select target language", options=list(LANGUAGES.keys()))

# Load the M2M100 model for translation
task_name = f"translation_{LANGUAGES[src_lang]}_to_{LANGUAGES[tgt_lang]}"
m2m100_translation = pipeline(task_name, model='facebook/m2m100_418M')

# Input text
input_text = st.text_area("Enter text to translate")

if st.button("Translate"):
    # Perform translation
    translated_texts = m2m100_translation(input_text)
    
    # Display translations
    st.subheader("Translations:")
    for translation in translated_texts:
        st.write(f"Input ({src_lang}): {input_text}")
        st.write(f"Translation ({tgt_lang}): {translation['translation_text']}\n")
