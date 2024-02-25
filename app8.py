import streamlit as st
from transformers import pipeline
import speech_recognition as sr

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

# Voice input
st.write("Speak now to provide input (or you can manually enter text)")
recording = st.checkbox("Start Recording")
if recording:
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)  # adjust for noise
        audio = recognizer.listen(source)  # listen for the audio via microphone
    input_text = recognizer.recognize_google(audio)  # recognize speech using Google Web Speech API
    st.write("Input Text:", input_text)

# Input text
input_text_manual = st.text_area("Or enter text to translate (optional)")

if st.button("Translate"):
    if recording:
        # If voice input was used, use the voice input text
        input_text = input_text
    else:
        # Otherwise, use manually entered text
        input_text = input_text_manual

    # Perform translation
    translated_texts = m2m100_translation(input_text)

    # Display translations
    st.subheader("Translations:")
    for translation in translated_texts:
        st.write(f"Input ({src_lang}): {input_text}")
        st.write(f"Translation ({tgt_lang}): {translation['translation_text']}\n")
