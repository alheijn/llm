def clean_text(text):
    import re
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def preprocess_dataset(dataset):
    import pandas as pd
    # Load the dataset
    df = pd.read_csv(dataset)
    # Apply cleaning to the text column (assuming the text column is named 'text')
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df[['cleaned_text']]  # Return only the cleaned text column