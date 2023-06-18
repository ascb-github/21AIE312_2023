import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pytesseract
import cv2
import os

# Title and Icon
# Emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Character Recognition", page_icon="📖", layout="wide")

# --- HEADER SECTION ---
# Center-align the title using Markdown and CSS
st.markdown(
    """
    <h1 style="text-align: center;">Character Recognition from Tamil Palm Leaves: Comparative Analysis of Different Classification Models</h1>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with st.container():
    st.write("---")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#MAKE CATEROGIRS
MAIN_DIR = "D:\\College_Semesters\\6th Semester\\C. Deep Learning for Signal & Image Processing\\Project_New\\Dataset"
category = os.listdir(MAIN_DIR)
#st.write("Categories: ",category)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Add custom CSS to style the file uploader
col1,col2,col3=st.columns(3)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# col1.markdown(
#     """
#     <style>
#     .small-file-uploader > label p {
#         font-size: 12px;
#         padding: 5px 10px;
#         background-color: #FFC0CB;
#         border-radius: 5px;
#         border: 1px solid #ccc;
#         line-height: 1;
#         margin: 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Add a file uploader widget with custom CSS class
# uploaded_file = col1.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png"],
#     key='file_uploader',
#     help="Only JPG, JPEG, and PNG files are allowed"
# )

# # Check if a file was uploaded
# if uploaded_file is not None:
#     # Read the contents of the file as bytes
#     image_bytes = uploaded_file.read()

#     # Display the uploaded image
#     col1.image(image_bytes, caption="Uploaded Image")


# Define the class labels
class_labels = ["ai", "cha", "ee", "ka", "la", "ma", "moo", "nna", "nnna", "nu", "nuu", "oo", "pa", "ra", "t", "tha", "va", "vee", "vu", "y", "ya"]

# Load the trained model
#model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\ResNet_3.h5')
base_model = tf.keras.applications.ResNet50(weights='imagenet')
model = tf.keras.models.Sequential([base_model, tf.keras.layers.Dense(len(class_labels), activation='softmax')])

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the class of the image
def predict_class(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions)
    class_name = class_labels[predicted_index]
    return class_name

def main():
    col2.title("Prediction")
    col2.write("Upload an image and let the app predict its class.")

    # File uploader
    uploaded_file = col2.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col2.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform the prediction
        with st.spinner('Predicting...'):
            predicted_class = predict_class(image)
            col2.success(f"Predicted Class: {predicted_class}")

# Run the app
if __name__ == '__main__':
    main()
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

col1.markdown(
    """
    <style>
    .small-file-uploader > label p {
        font-size: 12px;
        padding: 5px 10px;
        background-color: #FFC0CB;
        border-radius: 5px;
        border: 1px solid #ccc;
        line-height: 1;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define options for the dropdown
options = ["VGG-16", "VGG-19", "ResNet-50", "AlexNet", "DenseNet", "LeNet-5", "MobileNet", "CustomCNN"]

# Create a dropdown with a default value
selected_option = col1.selectbox("Select a model", options)

# # Display the selected option
# col2.text("You selected:", selected_option)

# Perform backend action based on selected option
if selected_option == "VGG-16":
    # Backend action 
    col1.write("Selected VGG-16")

elif selected_option == "VGG-19":
    # Backend action 
    col1.write("Selected VGG-19")

elif selected_option == "ResNet-50":
    # Backend action
    col1.write("Selected ResNet-50")

elif selected_option == "AlexNet":
    # Backend action 
    col1.write("Selected AlexNet")

elif selected_option == "DenseNet":
    # Backend action
    col1.write("Selected DenseNet")

elif selected_option == "LeNet-5":
    # Backend action 
    col1.write("Selected LeNet-5")

elif selected_option == "MobileNet":
    # Backend action
    col1.write("Selected MobileNet")

elif selected_option == "CustomCNN":
    # Backend action 
    col1.write("Selected CustomCNN")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

col3.markdown(
    """
    <style>
    .small-file-uploader > label p {
        font-size: 12px;
        padding: 5px 10px;
        background-color: #FFC0CB;
        border-radius: 5px;
        border: 1px solid #ccc;
        line-height: 1;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col3.write("Results")


if selected_option == "VGG-16":
    # Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\VGG16_1.h5')

    col3.write(f"Accuracy: 12.280701754385964")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\VGG16_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\VGG16_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)

    # # Function to preprocess the image
    # def preprocess_image(image):
    #     image = image.resize((224, 224))
    #     image = np.array(image) / 255.0
    #     image = np.expand_dims(image, axis=0)
    #     return image


    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = preprocess_image(img)

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label_index = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label_index]

    #     # # Display the predicted label and category
    #     # col3.write("Predicted Label: " + str(predicted_label))
    #     # col3.write("Predicted Category: " + predicted_category)

    #     # Display the uploaded image
    #     st.image(img, caption="Uploaded Image", use_column_width=True)

    #     # Display the predicted category
    #     st.write("Predicted Category:", predicted_category)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "VGG-19":
    # Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\VGG19_1.h5')

    col3.write(f"Accuracy: 12.280701754385964")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\VGG19_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\VGG19_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)

    # # Function to preprocess the image
    # def preprocess_image(image):
    #     image = image.resize((224, 224))
    #     image = np.array(image) / 255.0
    #     image = np.expand_dims(image, axis=0)
    #     return image


    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = preprocess_image(img)

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label_index = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label_index]

    #     # # Display the predicted label and category
    #     # col3.write("Predicted Label: " + str(predicted_label))
    #     # col3.write("Predicted Category: " + predicted_category)

    #     # Display the uploaded image
    #     #col3.image(img, caption="Uploaded Image", use_column_width=True)

    #     # Display the predicted category
    #     #col3.write("Predicted Category:", predicted_category)
    #     col3.write(f"Predicted Category: {predicted_category}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "ResNet-50":

    col3.write(f"Accuracy: 96.11111111111111")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\ResNet_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\ResNet_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)
            
    # # Load your pre-trained model
    # model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\ResNet_1.h5')

    # # Function to preprocess the image
    # def preprocess_image(image):
    #     image = image.resize((224, 224))
    #     image = np.array(image) / 255.0
    #     image = np.expand_dims(image, axis=0)
    #     return image


    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = preprocess_image(img)

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label_index = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label_index]

    #     # # Display the predicted label and category
    #     # col3.write("Predicted Label: " + str(predicted_label))
    #     # col3.write("Predicted Category: " + predicted_category)

    #     # Display the uploaded image
    #     st.image(img, caption="Uploaded Image", use_column_width=True)

    #     # Display the predicted category
    #     st.write("Predicted Category:", predicted_category)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "AlexNet":
    # Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\AlexNet_1.h5')

    col3.write(f"Accuracy: 9.62962962962963")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\AlexNet_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\AlexNet_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)

    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = img.resize((224, 224))  # Resize the image to match the model's input size
    #     img = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    #     img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label]

    #     # Display the predicted label and category
    #     col3.write("Predicted Label: " + str(predicted_label))
    #     col3.write("Predicted Category: " + predicted_category)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "DenseNet":
    # Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\DenseNet_1.h5')

    col3.write(f"Accuracy: 98.33333333333333")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\DenseNet_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\DenseNet_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)
    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = img.resize((224, 224))  # Resize the image to match the model's input size
    #     img = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    #     img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label]

    #     # Display the predicted label and category
    #     col3.write("Predicted Label: " + str(predicted_label))
    #     col3.write("Predicted Category: " + predicted_category)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "LeNet-5":
    # Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\LeNet_1.h5')

    col3.write(f"Accuracy: 87.96296296296296")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\LeNet_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\LeNet_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)

    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = img.resize((224, 224))  # Resize the image to match the model's input size
    #     img = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    #     img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label]

    #     # Display the predicted label and category
    #     col3.write("Predicted Label: " + str(predicted_label))
    #     col3.write("Predicted Category: " + predicted_category)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "MobileNet":
    # Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\MobileNet_1.h5')

    col3.write(f"Accuracy: 90.18518518518519")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\MobileNet_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\MobileNet_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)

    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = img.resize((224, 224))  # Resize the image to match the model's input size
    #     img = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    #     img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label]

    #     # Display the predicted label and category
    #     col3.write("Predicted Label: " + str(predicted_label))
    #     col3.write("Predicted Category: " + predicted_category)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

elif selected_option == "CustomCNN":
    ## Load your pre-trained model
    model = tf.keras.models.load_model('D:\\College_Semesters\\6th Semester\\Projects\\DL\\Codes\\Custom_1.h5')

    col3.write(f"Accuracy: 35.74074074074074")

    # Load the image
    image = Image.open("C:\\Users\\psadh\\Downloads\\DL\\CustomNet_1.png")

    # Display the image
    col3.image(image, caption='Training and Validation Accuracy', use_column_width=True)

    # Load the image
    image2 = Image.open("C:\\Users\\psadh\\Downloads\\DL\\CustomNet_2.png")

    # Display the image
    col3.image(image2, caption='Training and Validation loss', use_column_width=True)

    # if uploaded_file is not None:
    #     # Preprocess the uploaded image
    #     img = Image.open(uploaded_file)
    #     img = img.resize((224, 224))  # Resize the image to match the model's input size
    #     img = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    #     img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch

    #     # Make predictions
    #     predictions = model.predict(img)
    #     predicted_label = np.argmax(predictions[0])
    #     predicted_category = category[predicted_label]

    #     # Display the predicted label and category
    #     col3.write("Predicted Label: " + str(predicted_label))
    #     col3.write("Predicted Category: " + predicted_category)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
