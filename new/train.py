import os
import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from torch.optim import AdamW


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

"""
def tokenize_function(examples, tokenizer):
    # Tokenize both inputs separately
    inputs_snippet = tokenizer(examples['snippet'], truncation=True, padding="max_length", max_length=128)
    inputs_output = tokenizer(examples['output'], truncation=True, padding="max_length", max_length=128)

    return {"input_ids": inputs_snippet['input_ids'], "input_ids_output": inputs_output['input_ids'],
            "attention_mask": inputs_snippet['attention_mask'], "attention_mask_output": inputs_output['attention_mask']}
"""


"""
class EmbeddingDifferenceModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base", config=config)

    def forward(self, input_ids_snippet, attention_mask_snippet, input_ids_output, attention_mask_output):
        # Get embeddings for both inputs
        snippet_embeddings = self.model.roberta(input_ids_snippet, attention_mask=attention_mask_snippet)[0]
        output_embeddings = self.model.roberta(input_ids_output, attention_mask=attention_mask_output)[0]
        
        # Compute the difference between embeddings (using the mean of the token embeddings)
        snippet_embeddings = snippet_embeddings.mean(dim=1)  # Mean pooling over tokens
        output_embeddings = output_embeddings.mean(dim=1)    # Mean pooling over tokens
        
        # Calculate the difference between the embeddings
        embedding_diff = snippet_embeddings - output_embeddings
        
        # Pass the difference through the classifier
        logits = self.model.classifier(embedding_diff)
        
        return logits
"""


# Function to compute accuracy
def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(axis=-1)  # Get predicted class (for classification tasks)
    return {"accuracy": accuracy_score(labels, preds)}


# Function to initialize the model and tokenizer
def initialize_model_and_tokenizer(model_name, categories):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(categories))
    # model = EmbeddingDifferenceModel.from_pretrained("roberta-base", num_labels=len(categories))
    return model, tokenizer


# Function to configure LoRA
def configure_lora(model):
    lora_config = LoraConfig(
        r=32,  # Low-rank matrix dimension
        lora_alpha=64,  # Scaling factor
        target_modules=["self.query", "self.key", "self.value", "output.dense"],  # Modules to apply LoRA
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",  # Task type: Sequence Classification
    )
    return get_peft_model(model, lora_config)


# Function to initialize training arguments
def initialize_training_args():
    return TrainingArguments(
        output_dir='./results',             # output directory
        num_train_epochs=15,                 # number of training epochs
        per_device_train_batch_size=16,     # batch size for training
        per_device_eval_batch_size=64,      # batch size for evaluation
        evaluation_strategy="epoch",        # evaluate at the end of each epoch
        save_strategy="epoch",              # save model at the end of each epoch
        warmup_steps=500,                   # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                  # strength of weight decay
        logging_dir='./logs',               # directory for storing logs
        logging_steps=10,
        load_best_model_at_end=True,        # load the best model at the end
        metric_for_best_model="accuracy"   # specify which metric to track for best model
    )


# Function to initialize optimizer
def initialize_optimizer(model):
    return AdamW(model.parameters(), lr=5e-5)


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
    """
    tokenized_dataset.set_format("torch", columns=["input_ids_snippet", "attention_mask_snippet", 
                                               "input_ids_output", "attention_mask_output", "labels"])
    """
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Configure LoRA
    model = configure_lora(model)

    # Initialize training arguments
    training_args = initialize_training_args()

    # Initialize optimizer
    optimizer = initialize_optimizer(model)

    # Trainer setup with compute_metrics function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics   # Use the custom metric function
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

    # Save the model and tokenizer (optional)
    # model.save_pretrained("./fine_tuned_lora")
    # tokenizer.save_pretrained("./fine_tuned_lora")


if __name__ == "__main__":
    main()

