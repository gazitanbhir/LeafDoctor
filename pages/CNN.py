import os
import json
from PIL import Image
import streamlit_authenticator as stauth
import numpy as np
import tensorflow as tf
import streamlit as st
import time
import yaml
from yaml.loader import SafeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"trained_model/plant disease_98.72.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"classes.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(200, 200)):
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create a batch of 1 image
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction)
    classes = class_indices
    predicted_class = list(classes.keys())[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]
    return predicted_class, confidence_score

st.write("# Upload an Image for Disease Prediction and Treatment 👋")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        resized_img = image.resize((150, 150))
        st.image(resized_img)

        if st.button('Analyze Image'):
                # Preprocess the uploaded image and predict the class
                #prediction = predict_image_class(model, uploaded_image, class_indices)
                predicted_class, confidence_score = predict_image_class(model, uploaded_image, class_indices)
                topic = str(predicted_class)
                percentage = confidence_score*100
                
                if percentage>0:

                    col1, col2 = st.columns(2)

                    with col1:   
                        st.success(f'Prediction: {topic}')
                
                    with col2:   
                        st.info(f'Confidence: {percentage:.1f}%')

                    chat = ChatGroq(temperature=0, groq_api_key="your_api_key", model_name="mixtral-8x7b-32768")
                    prompt = ChatPromptTemplate.from_messages([("human", "What comprehensive steps can we take for {topic} in our plants? A brief, step-by-step guide with additional tips would be greatly appreciated. Don't use words like I,We,Me. Instantly start with guide, no other talks, write it very precisely within 500 words ")])
                    chain = prompt | chat
                    my_string = ""
                    for chunk in chain.stream({"topic": {topic}}):
                        print(chunk.content, end="", flush=True)
                        my_string += chunk.content + ""  
                
                    bar = st.progress(20)
                    time.sleep(1)
                    bar.progress(100)

                    st.write(my_string)


                else:
                     st.error('The uploaded image is invalid or lacks sufficient features for analysis')
                

