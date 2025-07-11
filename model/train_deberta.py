import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DebertaV2Tokenizer
from datasets import Dataset
import transformers
import os

print("Transformers version:", transformers.__version__)
print("Transformers file:", transformers.__file__)
print("TrainingArguments:", TrainingArguments)
print("TrainingArguments file:", TrainingArguments.__module__)

# Check device availability
device = torch.device('cpu')
print(f"Using device: {device}")

# 1. Load the separate datasets (no splitting needed)
TRAIN_PATH = '../data/train_data.csv'
VAL_PATH = '../data/val_data.csv'
TEST_PATH = '../data/test_data.csv'

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train dataset shape: {train_df.shape}")
print(f"Validation dataset shape: {val_df.shape}")
print(f"Test dataset shape: {test_df.shape}")

print(f"Train class distribution:")
print(train_df['fraudulent'].value_counts().sort_index())
print(f"Validation class distribution:")
print(val_df['fraudulent'].value_counts().sort_index())
print(f"Test class distribution:")
print(test_df['fraudulent'].value_counts().sort_index())

# 2. Prepare text and label columns for each dataset
def combine_text(row):
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    req = str(row['requirements']) if pd.notnull(row['requirements']) else ''
    return desc + ' ' + req

train_df['text'] = train_df.apply(combine_text, axis=1)
val_df['text'] = val_df.apply(combine_text, axis=1)
test_df['text'] = test_df.apply(combine_text, axis=1)

train_df = train_df[['text', 'fraudulent']]
val_df = val_df[['text', 'fraudulent']]
test_df = test_df[['text', 'fraudulent']]

# 3. Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# Ensure labels are binary (0 or 1) and have no NaNs
def convert_to_binary(x):
    # Convert any non-zero value to 1, zero or negative to 0
    return {'labels': 1 if x['fraudulent'] > 0 else 0}

train_dataset = train_dataset.map(convert_to_binary)
val_dataset = val_dataset.map(convert_to_binary)
test_dataset = test_dataset.map(convert_to_binary)

# Debug: Check dataset features
print("Train dataset features:", train_dataset.features)
print("Validation dataset features:", val_dataset.features)
print("Test dataset features:", test_dataset.features)
print("Sample train labels:", train_dataset['labels'][:5])
print("Sample validation labels:", val_dataset['labels'][:5])
print("Sample test labels:", test_dataset['labels'][:5])
print("Train label distribution:", train_dataset['labels'].count(0), "zeros,", train_dataset['labels'].count(1), "ones")
print("Validation label distribution:", val_dataset['labels'].count(0), "zeros,", val_dataset['labels'].count(1), "ones")
print("Test label distribution:", test_dataset['labels'].count(0), "zeros,", test_dataset['labels'].count(1), "ones")

# 4. Load DeBERTa tokenizer and model
MODEL_NAME = 'microsoft/deberta-v3-base'
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# 5. Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch with 'labels'
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 6. Load model for binary classification
MODEL_LOCAL_PATH = './model/deberta_best_model'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_LOCAL_PATH)

# Move model to device
model = model.to(device)

# Debug: Check if parameters require gradients
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Model device: {next(model.parameters()).device}")

# 7. Training arguments
training_args = TrainingArguments(
    output_dir='./model/deberta_results',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./model/deberta_logs',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to='none',
    # Add device configuration
    no_cuda=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0)
    }

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train the model
print("Starting training...")
trainer.train()

# 10. Evaluate and print metrics
print("Validation metrics:")
val_metrics = trainer.evaluate()
for k, v in val_metrics.items():
    print(f'{k}: {v}')

# Print validation classification report
val_preds_output = trainer.predict(val_dataset)
val_preds = np.argmax(val_preds_output.predictions, axis=1)
print('\nValidation classification report:')
print(classification_report(val_preds_output.label_ids, val_preds, digits=4))

# Final evaluation on test set
print("\nTest metrics:")
test_preds_output = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_output.predictions, axis=1)

# Calculate test metrics manually
test_accuracy = accuracy_score(test_preds_output.label_ids, test_preds)
test_precision = precision_score(test_preds_output.label_ids, test_preds, zero_division=0)
test_recall = recall_score(test_preds_output.label_ids, test_preds, zero_division=0)
test_f1 = f1_score(test_preds_output.label_ids, test_preds, zero_division=0)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1: {test_f1:.4f}')

print('\nTest classification report:')
print(classification_report(test_preds_output.label_ids, test_preds, digits=4))

# 11. Save the best model
trainer.save_model('./model/deberta_best_model')

# Optional: To visualize training and validation loss, run this in your terminal after training:
# tensorboard --logdir=./model/deberta_logs


print(TrainingArguments)
args = TrainingArguments(output_dir='.', eval_strategy='epoch')
print(args) 
