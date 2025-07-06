
from transformers import TrainingArguments
print(TrainingArguments)
args = TrainingArguments(output_dir='.', eval_strategy='epoch')
print(args)