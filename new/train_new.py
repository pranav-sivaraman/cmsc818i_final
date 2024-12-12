import os
import json
import random
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F

os.environ["WANDB_DISABLED"] = "true"


# Function to load and shuffle dataset
def load_and_shuffle_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    return data


# Function to convert JSON data to Hugging Face Dataset
def convert_to_dataset(data):
    return Dataset.from_list(data)


# Function to map categories to integers
def map_labels(example, category_to_id):
    example['label'] = category_to_id[example['category']]
    return example


# Function to tokenize the dataset
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['snippet'],  # Original text
        examples['output'],   # Copied/non-copied text
        truncation=True,
        padding="max_length",
        max_length=128
    )


# Function to compute accuracy
def compute_metrics(preds, labels):
    preds = preds.argmax(axis=-1)  # Get predicted class (for classification tasks)
    return {"accuracy": accuracy_score(labels, preds)}


# Function to initialize the model and tokenizer
def initialize_model_and_tokenizer(model_name, categories):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(categories))
    return model, tokenizer


# Define the model that takes the difference of embeddings
class EmbeddingDifferenceModel(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingDifferenceModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 2)  # 2 classes (copied, non-copied)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        pooled_embedding = embeddings.mean(dim=1)  # Mean pooling (shape: batch_size, hidden_size)
        # Compute the embedding difference with mean pooling (element-wise difference)
        pooled_embedding = pooled_embedding.view(pooled_embedding.size(0), -1)
        logits = self.classifier(pooled_embedding)
        return logits


# Function to initialize training arguments
def initialize_training_args():
    return {
        'epochs': 6,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 500
    }


# Main training loop
def main():
    # Load and prepare dataset
    data = load_and_shuffle_dataset("dataset.json")
    dataset = convert_to_dataset(data)

    # Split into train and test sets
    dataset = dataset.train_test_split(test_size=0.2)

    # Map categories to integers
    categories = list(set([item['category'] for item in data]))
    category_to_id = {category: idx for idx, category in enumerate(categories)}
    dataset = dataset.map(lambda example: map_labels(example, category_to_id))

    # Tokenize dataset
    model_name = 'roberta-base'
    model, tokenizer = initialize_model_and_tokenizer(model_name, categories)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Prepare for PyTorch
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Initialize model with custom embedding difference approach
    model = EmbeddingDifferenceModel(model)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Training loop
    training_args = initialize_training_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(training_args['epochs']):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in tokenized_dataset['train']:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        # Compute accuracy and loss for the epoch
        epoch_accuracy = correct_predictions / total_predictions
        epoch_loss = total_loss / len(tokenized_dataset['train'])

        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")

        # Evaluation after each epoch
        model.eval()
        eval_loss = 0
        eval_correct = 0
        eval_total = 0

        for batch in tokenized_dataset['test']:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs, labels)
                eval_loss += loss.item()

                preds = outputs.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)

        eval_accuracy = eval_correct / eval_total
        eval_loss = eval_loss / len(tokenized_dataset['test'])

        print(f"Validation Loss = {eval_loss:.4f}, Validation Accuracy = {eval_accuracy:.4f}")


if __name__ == "__main__":
    main()

