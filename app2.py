from transformers import pipeline

# Load the M2M100 model
m2m100 = pipeline('translation', model='facebook/m2m100_418M', src_lang='en', tgt_lang='fr')

input_text = ["Hello, world!", "This is amazing!"]
translated_texts = m2m100(input_text)

for translation in translated_texts:
    # Check for the correct key name (e.g., 'source_text' instead of 'input_text')
    if "source_text" in translation:
        original_text = translation["source_text"]
    elif "input_ids" in translation:
        # Decode from token IDs if necessary
        original_text = m2m100.tokenizer.decode(translation["input_ids"], skip_special_tokens=True)
    else:
        # Handle case where input text is missing
        original_text = "Error: Input text not found in output"

    print(f"Input: {original_text}")
    print(f"Translation: {translation['translation_text']}\n")

# If translating to Spanish, adapt the code similarly
m2m100_es = pipeline('translation', model='facebook/m2m100_418M', src_lang='de', tgt_lang='es')
# ... proceed with translation and access input text as needed

