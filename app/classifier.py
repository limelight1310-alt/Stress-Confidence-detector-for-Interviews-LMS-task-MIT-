from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import cv2
import numpy as np

print("Loading emotion model...")
model_name = "trpakov/vit-face-expression"
extractor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()
print("Model loaded successfully")

def classify_emotion(face_frame):

    if face_frame is None or face_frame.size == 0:
        return {
            'neutral': 100,
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'fear': 0,
            'disgust': 0,
            'surprise': 0
        }

    face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)
    inputs = extractor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    probs = probabilities[0].tolist()

    emotions = {}
    for i, prob in enumerate(probs):
        if i < len(model.config.id2label):
            label = model.config.id2label[i].lower()
            emotions[label] = round(prob * 100, 2)

    return emotions