from transformers import pipeline

# Load pre-trained model from Hugging Face
hate_speech_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-hatespeech")

# Function to classify input text
def classify_hate_speech(text):
    result = hate_speech_model(text)
    return result

if __name__ == "__main__":
    # Example input text
    input_text = input("Enter text for classification: ")
    classification_result = classify_hate_speech(input_text)

    # Output the result
    print(f"Result: {classification_result[0]['label']}, Confidence: {classification_result[0]['score']}")
