import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)        # remove links
    text = re.sub(r'[^a-zA-Z ]', '', text)    # remove symbols
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text.strip()
