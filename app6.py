import streamlit as st
from transformers import pipeline

# Load the M2M100 model for English to Hindi translation
m2m100_en_to_hi = pipeline('translation', model='facebook/m2m100_418M', src_lang='en', tgt_lang='hi')

st.title("English to Hindi Translator")

# Input English text
input_text = st.text_area("Enter English text to translate")

if st.button("Translate"):
    # Perform translation
    translated_texts = m2m100_en_to_hi(input_text)
    
    # Display translations
    st.subheader("Translations:")
    for translation in translated_texts:
        st.write(f"Input: {input_text}")
        st.write(f"Translation: {translation['translation_text']}\n")
