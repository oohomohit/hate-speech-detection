from transformers import pipeline
import pandas as pd

# Load the model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to classify text
def classify_text(text):
    candidate_labels = ["Religious Hate Speech", "Neutral", "Political Debate"]
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # Return the most likely label

# Load the tweet dataset
df = pd.read_csv("religious_hate_speech_tweets.csv")

# Apply classification to all tweets in the dataset
df['classification'] = df['Text'].apply(classify_text)

# Save the classified data to a new CSV file
df.to_csv("classified_tweets.csv", index=False)
print("Classified dataset saved!")
