from statistics import mode
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
import os
os.environ['WANDB_SILENT'] = 'true'
wandb.disabled = True
wandb.init(mode="disabled")

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(1000))

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
trainings_args = TrainingArguments(output_dir='test_trainer', evaluation_strategy='epoch')
trainer = Trainer(model=model,
                  args=trainings_args,
                  train_dataset=small_train_dataset,
                  eval_dataset=small_eval_dataset,
                  compute_metrics=compute_metrics)
trainer.train()
# print('model', model)

# del model
# del trainer
# torch.cuda.empty_cache()