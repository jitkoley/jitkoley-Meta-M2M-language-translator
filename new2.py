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

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        st.write("Processing...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")

# Function to perform translation
def perform_translation(input_text):
    translated_texts = m2m100_translation(input_text)
    return translated_texts

# Text input box for keyboard input
user_input_text = st.text_input("Enter text:", "")

# Define voice_input_text outside of the button callback
voice_input_text = None

# Voice input button for microphone input
voice_button_col, _ = st.columns([1, 20])
with voice_button_col:
    if st.button("Speak", help="Click to Start Recording (Voice Input)"):
        voice_input_text = recognize_speech()
        st.write("Voice Input:", voice_input_text)

# Perform translation if either voice or keyboard input is obtained
if user_input_text or voice_input_text:
    if user_input_text:
        input_text = user_input_text
    else:
        input_text = voice_input_text

    translated_texts = perform_translation(input_text)
    st.subheader("Translations:")
    for translation in translated_texts:
        st.write(f"Input: {input_text}")
        st.write(f"Translation ({tgt_lang}): {translation['translation_text']}\n")
