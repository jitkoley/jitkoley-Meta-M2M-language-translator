from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the pre-trained model
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

# Load the tokenizer
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="hi")


src_text = "UN Chief Says There Is No Military Solution in Syria"
translated_text = model.generate(tokenizer.encode(src_text, return_tensors="pt"))
decoded_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)
print(f"Translated text (Romanian): {decoded_text}")
