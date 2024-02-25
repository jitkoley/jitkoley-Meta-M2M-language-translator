from transformers import pipeline

# Load the M2M100 model for English to Bengali translation
m2m100_en_to_bn = pipeline('translation', model='facebook/m2m100_418M', src_lang='en', tgt_lang='bn')

# Input English text
input_text = ["Hello!! good morning"]

# Perform translation
translated_texts = m2m100_en_to_bn(input_text)

# Display translations
for translation in translated_texts:
    print(f"Input: {translation['input_text']}")
    print(f"Translation: {translation['translation_text']}\n")
