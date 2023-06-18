import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.title("Object Detection using Robot Captured Images")
image = Image.open("D:\\Amrita\\Programming\\Python\\DeepLearning\\Project\\MobileRobot.jpg")
st.image(image, width=700)

model = load_model("D:\\Amrita\\Programming\\Python\\DeepLearning\\Project\\model.h5")

class_names = ["Bottle", "Bowl", "CorkScrew", "Cottonswab", "Cup", "Cushion", "Knife"]

def normalize_image(image):
    image = image / 255.0
    return image

def predict_class(image):
    target_size = (170, 170)
    input_image = cv2.resize(image, target_size)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = normalize_image(input_image)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions[0])
    return class_names[predicted_class_index]

def add_boundary_box(image):
    top_left = (40, 40)
    bottom_right = (image.shape[1] - 40, image.shape[0] - 40)
    box_color = (0, 255, 0)
    box_thickness = 2
    image_with_box = cv2.rectangle(image, top_left, bottom_right, box_color, box_thickness)
    mask = np.zeros_like(image)
    mask = cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)
    image_with_mask = cv2.addWeighted(image_with_box, 0.7, mask, 0.3, 0)
    return image_with_mask

def main():
    option = st.selectbox("Select an option", ["Choose an option", "Object Detection", "Real Time Object Detection"])

    if option == "Choose an option":
        st.write("Please select an option from the dropdown menu.")

    elif option == "Object Detection":
        st.write("Upload an image and click the 'Predict' button to classify the object and generate a mask around the object.")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            predicted_class = predict_class(image)
            image_with_mask = add_boundary_box(image)
            image_rgb = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_column_width=True)
            st.write("Predicted class:", predicted_class)
    
    elif option == "Real Time Object Detection":
        st.write("Click the link to control the robot and view the real time object detection.")
        st.markdown("[OpenAI](https://openai.com)")
        
if __name__ == "__main__":
    main()
