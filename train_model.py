# train_model.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import pandas as pd

# Load preprocessed data
def load_data(file_path):
    """
    Load preprocessed data (CSV format)
    """
    return pd.read_csv(file_path)

# Create a DataLoader for model training
def create_dataloader(data, batch_size=16):
    """
    Create a PyTorch DataLoader from processed data.
    """
    input_ids = torch.tensor(data['input_ids'].tolist())
    attention_mask = torch.tensor(data['attention_mask'].tolist())
    labels = torch.tensor(data['label'].tolist())  # Assuming 'label' column exists in the data
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size)

# Define the model
def define_model(num_labels=2):
    """
    Load a pre-trained BERT model for sequence classification.
    """
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    return model

# Training function using Trainer API
def train_model(model, train_dataloader, eval_dataloader):
    """
    Train the model using Hugging Face Trainer API.
    """
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
    )
    
    trainer.train()

# Main function
def main():
    # Load preprocessed data
    data = load_data("data/processed/preprocessed_data.csv")
    
    # Split the data into training and evaluation sets
    train_data, eval_data = train_test_split(data, test_size=0.1)
    
    # Create dataloaders for training and evaluation
    train_dataloader = create_dataloader(train_data)
    eval_dataloader = create_dataloader(eval_data)
    
    # Define the model
    model = define_model(num_labels=2)  # Assuming binary classification (hate speech or not)
    
    # Train the model
    train_model(model, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()
