from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor
from io import BytesIO

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

key_map = {
    "akiec":"Actinic Keratoses",
    "bcc":"Basal Cell Carcinoma",
    "bkl":"Benign Keratoses-Like Lesions",
    "df":"Dermatofibroma",
    "mel":"Melanoma",
    "nv" :"Melanocytic Nevi",
    "vasc" :"Vascular Lesions",
}

binary_model = models.resnet18(weights=None)
num_features = binary_model.fc.in_features
binary_model.fc = torch.nn.Linear(num_features, 2)
checkpoint_path = "./binary_classifier_model/best_resnet18_binary.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
binary_model.load_state_dict(checkpoint["model"])
binary_model = binary_model.to(device) #assignment might be unnecssary
binary_model.eval()

binary_labels = {0: "non_lesion", 1: "lesion"}

binary_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

lesion_model = AutoModelForImageClassification.from_pretrained("./skin_model/checkpoint-6762")
lesion_processor = AutoImageProcessor.from_pretrained("./skin_model/checkpoint-6762")

labels = lesion_model.config.id2label

print("M1: ")
print(lesion_model.config.id2label)
print(lesion_model.config.label2id)

lesion_model.eval()

app = FastAPI()

# Add CORS (Cross-Origin Resource Sharing) so your React Native app can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# This route is just a test
@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}


# This route will handle image uploads from your React Native app
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()  # Read the uploaded file contents into memory

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img_tensor = binary_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        binary_logits = binary_model(img_tensor)
        binary_probs = F.softmax(binary_logits, dim=1)[0] #tensor wrapped array of probabilities
        binary_pred = torch.argmax(binary_probs).item() #gets predicted index (0: non lesion, 1: lesion)
    
    #dict with key/value pair like 'non_lesion: prediction'
    binary_result  = { 
        binary_labels[i]: (round(prob.item(), 4))
        for i, prob in enumerate(binary_probs)
    }

    inputs = lesion_processor(images = image, return_tensors="pt")
    with torch.no_grad():
        outputs = lesion_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

    probabilities = {
        labels[i]: round(prob.item(), 4)
        for i, prob in enumerate(probs)
    }

    probabilities = {key_map[key]: value for key, value in probabilities.items()}

    return {
        "probabilities": binary_result,
        "predicted_class": binary_labels[binary_pred],
        "binary_pred": binary_pred,
        "confidence:": round(torch.max(binary_probs).item(), 4),
        "confidence_boolean": binary_probs[1].item() > 0.85
    }
    
@app.post("/full_classify/")
async def full_classify(file: UploadFile = File(...)):
    contents = await file.read()
    
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = lesion_processor(images = image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = lesion_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]
        
    probabilities = {
        labels[i]: round(prob.item(), 4)
        for i, prob in enumerate(probs)
    }

    probabilities = {key_map[key]: value for key, value in probabilities.items()}

    print("\n--- Model 1 Probabilities ---")
    for label, score in probabilities.items():
        print(f"{label}: {score}")

    return {
        "key_map": key_map,
        "filename": file.filename,
        "probabilities": probabilities,
        "predicted_class": key_map[labels[torch.argmax(probs).item()]],
        "lesion_confidence": round(torch.max(probs).item(), 4),
    }