import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import time


import torch
from torchvision.models import resnet50
from torch import nn
from torchvision import transforms
from PIL import Image

with open('/home/hydra/Documents/Streamlit/app/class_labels.json', 'r') as file:
    class_labels = json.load(file)

path_model = "/home/hydra/Documents/Streamlit/app/trained_model/01_plant_diseases_classification_pytorch_rn50.pth"
model = resnet50(weights=None)
model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features, out_features=38))
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
model.eval()

# define the transfomation for the test images (same as transformation for validation data)
preprocess = transforms.Compose([
    transforms.Resize(size=232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=224),
    # do not augment in test data
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_path = 'http://localhost:8501/media/8e6b115d725311c2f1edf485800d5f7cdcc3616f05acdfcc6b8b45b5.jpg'
# prediction function on single image
def predict_image(image_path, model):
    # open and preprocess the image
    image = Image.open(image_path)
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

    # make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        predicted_label = class_labels[predicted.item()]
        conf = confidence[predicted.item()]
    return predicted_label, confidence, conf

# Streamlit App
st.title('Automated Crop Health Analysis System')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_path = 'http://localhost:8501/media/8e6b115d725311c2f1edf485800d5f7cdcc3616f05acdfcc6b8b45b5.jpg'
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify resnet'):
            # Preprocess the uploaded image and predict the class
            predicted_label, confidence, conf = predict_image(image_path, model)
            st.success(f'Prediction:  {predicted_label}')
            st.success(f'Prediction Confidence: {conf}')






