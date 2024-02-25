from transformers import pipeline

# Load the M2M100 model for English to Hindi translation
m2m100_en_to_hi = pipeline('translation', model='facebook/m2m100_418M', src_lang='en', tgt_lang='hi')

# Input English text
input_text = ["Hello, world!", "This is amazing!"]

# Perform translation
translated_texts = m2m100_en_to_hi(input_text)

# Display translations
for translation in translated_texts:
    print(translation)  # Print the entire response dictionary
    print(f"Translation: {translation['translation_text']}\n")
