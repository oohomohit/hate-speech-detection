# preprocess.py
import pandas as pd
import re
from transformers import BertTokenizer

# Load your dataset
def load_data(file_path):
    """
    Load dataset (Assuming CSV format for example)
    """
    return pd.read_csv(file_path)

# Clean and preprocess the text data
def clean_text(text):
    """
    Remove URLs, mentions, hashtags, special characters, etc.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions (e.g., @user)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters, digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Lowercase text
    text = text.lower()
    
    return text

# Tokenize the text using BERT tokenizer
def tokenize_text(text, tokenizer, max_length=128):
    """
    Tokenize the text using the specified tokenizer.
    """
    return tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

# Preprocess the entire dataset
def preprocess_data(input_path, output_path, tokenizer):
    """
    Preprocess the dataset and save the preprocessed file.
    """
    # Load data
    data = load_data(input_path)
    
    # Clean the text data
    data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x))
    
    # Tokenize the text
    data['input_ids'] = data['cleaned_text'].apply(lambda x: tokenize_text(x, tokenizer)['input_ids'])
    data['attention_mask'] = data['cleaned_text'].apply(lambda x: tokenize_text(x, tokenizer)['attention_mask'])

    # Save preprocessed data
    data.to_csv(output_path, index=False)

# Main function
def main():
    # Define the tokenizer (using BERT here)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Input and Output file paths
    input_path = "data/raw/hate_speech_data.csv"  # Change this path to your dataset file
    output_path = "data/processed/preprocessed_data.csv"
    
    # Run the preprocessing
    preprocess_data(input_path, output_path, tokenizer)

if __name__ == "__main__":
    main()
