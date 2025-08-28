import pandas as pd
from PIL import Image
import torch
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
import torch.nn.functional as F


model = AutoModelForImageClassification.from_pretrained("../skin_model/checkpoint-6762")
processor = AutoImageProcessor.from_pretrained("../skin_model/checkpoint-6762")

#model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification") ##
#processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification") ##

print(model.config.id2label)
print(model.config.label2id)
print(model.config.num_labels)

image_path = "D:/Downloads/bcc.jpeg"

model.eval() 
image = Image.open(image_path).convert("RGB")

# Apply processor transforms
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]

id2label = model.config.id2label

print(f"Class probability breakdown for: {image_path}\n")
for idx, prob in enumerate(probs):
    label = id2label[idx]
    percent = prob.item() * 100
    print(f"{label:5s}: {percent:.2f}%")


# Load metadata CSV again
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS, 
    "kmader/skin-cancer-mnist-ham10000", 
    "HAM10000_metadata.csv"
)

base_path = "D:\\Downloads"

# Fix image paths
def get_image_path(image_id):
    path1 = os.path.join(base_path, "HAM10000_images_part_1", f"{image_id}.jpg")
    path2 = os.path.join(base_path, "HAM10000_images_part_2", f"{image_id}.jpg")
    return path1 if os.path.exists(path1) else path2

df["image_path"] = df["image_id"].apply(get_image_path)

# Map labels
id2label = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
#id2label = {2: 'akiec', 1: 'bcc', 0: 'bkl', 6: 'df', 5: 'mel', 4: 'nv', 3: 'vasc'}
label2id = {v: k for k, v in id2label.items()}
#print(df[["dx"]][:10])
#print(df.columns)
#print("SECOND PART: ")
df["label"] = df["dx"].map(label2id) ## model.config.label2id
mel_df = df[df['dx'] == 'bcc']
print(df.head())
print(mel_df.head())
"""
# Create Hugging Face dataset
hf_dataset = Dataset.from_pandas(df[["image_path", "label"]])

def transform(example):
    image = Image.open(example["image_path"]).convert("RGB")
    processed = processor(images=image, return_tensors="pt")
    example["pixel_values"] = processed["pixel_values"][0]
    return example

hf_dataset = hf_dataset.map(transform)
eval_dataset = hf_dataset

def compute_metrics(p):
    print("in")
    preds = p.predictions.argmax(-1)
    print(f"Labels: {p.label_ids[:5]}, Preds: {preds[:5]}")
    return {"accuracy": accuracy_score(p.label_ids, preds)}

args = TrainingArguments(
    output_dir="./dummy_eval",
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    do_train=False
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

print("starting evaluation")
metrics = trainer.evaluate()
print(metrics)
print(f"Checkpoint 6762 Accuracy: {metrics['eval_accuracy']:.4f}")
"""