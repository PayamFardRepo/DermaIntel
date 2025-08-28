import pandas as pd
from PIL import Image
import os
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 1. Authenticate KaggleHub
model = AutoModelForImageClassification.from_pretrained(
    "Anwarkh1/Skin_Cancer-Image_Classification"
)

print(model.config.id2label)

# 2. Load CSV from Kaggle
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS, 
    "kmader/skin-cancer-mnist-ham10000", 
    "HAM10000_metadata.csv"
)

base_path = "D:\\Downloads"

# 3. Add image path column from the dataset
# Note: Image folders are split into two parts in the Kaggle dataset

def get_image_path(image_id): 
    path1 = os.path.join(base_path, "HAM10000_images_part_1", f"{image_id}.jpg")
    path2 = os.path.join(base_path, "HAM10000_images_part_2", f"{image_id}.jpg")
    return path1 if os.path.exists(path1) else path2

df["image_path2"] = df["image_id"].apply(get_image_path) # <- applys image path for each jpg

id2label = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
label2id = {v: k for k, v in id2label.items()}
df["label"] = df["dx"].map(label2id) # maps dx labels to numbers for model

hf_dataset = Dataset.from_pandas(df[["image_path2", "label"]])
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

def transform(example):
    image = Image.open(example["image_path2"]).convert("RGB")
    image_processed = processor(images=image, return_tensors="pt")
    example["pixel_values"] = image_processed["pixel_values"][0]
    return example

hf_dataset = hf_dataset.map(transform)

# 8. Train/test split
split_dataset = hf_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 9. Load model
model = AutoModelForImageClassification.from_pretrained(
    "Anwarkh1/Skin_Cancer-Image_Classification",
    num_labels=7,
    id2label=id2label,
    label2id=label2id
)

# 10. Training arguments
args = TrainingArguments(
    output_dir="./skin_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False
)

# 11. Accuracy metric
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# 12. Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# 13. Start training
trainer.train()

metrics = trainer.evaluate()
print(f"Final accuracy: {metrics['eval_accuracy']}")




"""
base_path = kagglehub.load_dataset(
    KaggleDatasetAdapter.FILES,
    "kmader/skin-cancer-mnist-ham10000"
)
def get_image_path(image_id):
    path1 = os.path.join(base_path, "HAM10000_images_part_1", f"{image_id}.jpg")
    path2 = os.path.join(base_path, "HAM10000_images_part_2", f"{image_id}.jpg")
    return path1 if os.path.exists(path1) else path2

df["image_path"] = df["image_id"].apply(get_image_path)

# 4. Create label mappings
id2label = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
label2id = {v: k for k, v in id2label.items()}
df["label"] = df["dx"].map(label2id)

# 5. Convert to Hugging Face dataset
hf_dataset = Dataset.from_pandas(df[["image_path", "label"]])

# 6. Load processor
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

# 7. Image transform function
def transform(example):
    image = Image.open(example["image_path"]).convert("RGB")
    image_processed = processor(images=image, return_tensors="pt")
    example["pixel_values"] = image_processed["pixel_values"][0]
    return example

hf_dataset = hf_dataset.map(transform)

# 8. Train/test split
split_dataset = hf_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 9. Load model
model = AutoModelForImageClassification.from_pretrained(
    "Anwarkh1/Skin_Cancer-Image_Classification",
    num_labels=7,
    id2label=id2label,
    label2id=label2id
)

# 10. Training arguments
args = TrainingArguments(
    output_dir="./skin_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False
)

# 11. Accuracy metric
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# 12. Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# 13. Start training
trainer.train()
"""