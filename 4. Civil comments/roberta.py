import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import datasets
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Nacitanie modelu
model_name = "FacebookAI/roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    problem_type="multi_label_classification"
)

# Nacitanie datasetu
dataset = load_dataset("data", data_files={"train": "train.csv"})
df = dataset["train"].to_pandas()

# Rozdelenie datasetu na train a test
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = datasets.Dataset.from_pandas(train_df)
test_dataset = datasets.Dataset.from_pandas(val_df)
dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})


# Tokenizacia datasetu do vektorovej podoby
def tokenize_and_encode(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    label_names = ["toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    labels = [examples.get(label_name, 0.0) for label_name in label_names]
    encoding["labels"] = torch.tensor(labels, dtype=torch.float)
    return encoding


tokenized_datasets = dataset.map(tokenize_and_encode, batched=False)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Nastavenie argumentov trenovania
training_args = TrainingArguments(
    output_dir="./RoBERTa",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,
)

# Nacitanie metrik pre evaluaciu
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


# Funkcia pre vypocet metrik
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (predictions > 0.5).astype(int)

    precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
    }


# Inicializacia
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# Spustenie trenovania modelu
trainer.train()

# Vykreslenie loss funkcie pocas celeho priebehu trenovania
loss_values = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = list(range(1, len(loss_values) + 1))

plt.figure(figsize=(10, 5))
plt.plot(steps, loss_values, marker='o', linestyle='-', color='b', label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid()
plt.show()
