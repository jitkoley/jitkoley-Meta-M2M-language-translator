from transformers import pipeline

# Load the M2M100 model
m2m100 = pipeline('translation', model='facebook/m2m100_418M', src_lang='en', tgt_lang='fr')

input_text = ["Hello, world!", "This is amazing!"]
translated_texts = m2m100(input_text)

for translation in translated_texts:
    print(translation)  # Print out the entire dictionary to see its structure
    print(f"Translation: {translation['translation_text']}\n")
