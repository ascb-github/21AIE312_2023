import streamlit as st
import networkx as nx
import requests
from matplotlib import cm
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import os
import tempfile
from tensorflow import keras
from keras.utils import img_to_array,load_img
import warnings
warnings.filterwarnings("ignore")
import visualkeras
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from streamlit.components.v1 import html

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('img.png') 


def sidebar_bg(side_bg):
   side_bg_ext = 'png'
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'mona1.png'
sidebar_bg(side_bg)

st.sidebar.title("Let's Get Started")

st.markdown("<h1 style='text-align:center; color: white;'>Prediction Of Alzheimers Disease</h1>",unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
model = keras.models.load_model("Final_Model_DL.h5")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if(uploaded_file == None):
    st.sidebar.write("Please upload the file")
    st.markdown("<h3 style='text-align:center; color: white;'>CNN Architecture</h3>",unsafe_allow_html=True)
    st.image(visualkeras.layered_view(model))
else:
    st.markdown("<h3 style='text-align:center; color: white;'>CNN Architecture</h3>",unsafe_allow_html=True)
    st.image(visualkeras.layered_view(model))
    img_width, img_height = 128, 128
    train_data_dir = '/Users/sreecharan/Desktop/University/Sem-6/DL_SIP/Alzheimer_s Dataset all/train'
    test_data_dir = '/Users/sreecharan/Desktop/University/Sem-6/DL_SIP/Alzheimer_s Dataset all/test'
    epochs = 7
    batch_size = 50
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    def read_image(file_path):
        print("[INFO] loading and preprocessing image…")
        image = load_img(file_path, target_size=(128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.
        return image

    def predict_proba(number):
        return [number[0],1-number[0]]


    def test_single_image(path):
        labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        image = read_image(path)
        bt_prediction = model.predict(image)
        preds = predict_proba(bt_prediction)
        st.markdown("<h3 style='text-align:center; color: white;'>Prediction</h3>",unsafe_allow_html=True)
        for idx, label, x in zip(range(0, 5), labels, preds[0]):
            st.markdown("ID: {}, Label: {} {}%".format(idx, label, np.round(x * 100, 2)))

        print("\n")
        print('----------------------------Final Decision----------------------------------')

        class_predicted = model.predict(image)
        class_predict=np.argmax(class_predicted,axis=1)
        class_dictionary = generator_top.class_indices
        inv_map = {v: k for k, v in class_dictionary.items()}
        print("ID: {}, Label: {}".format(class_predict[0], inv_map[class_predict[0]]))
        return inv_map[class_predict[0]]

#-------------------------------------------------------------------------------------------
    vizu = st.sidebar.selectbox("Do you want to see layers of the Architecture?",("None","Yes","No"))
    if(vizu=="Yes"):
        st.markdown("<h3 style='text-align:center; color: white;'>Layered Architecture</h3>",unsafe_allow_html=True)
        layers_img=load_img("model.png")
        left_co, cent_co,last_co = st.columns(3)
        with cent_co: 
            st.image(layers_img)
    Predicted_Class=test_single_image(uploaded_file)
    st.markdown("<h5>Predicted Class:</h5>",unsafe_allow_html=True)
    st.markdown(Predicted_Class)
    st.image(uploaded_file)

    #---------------------------------------------------------------------------------------  

    class GradCAM:
        def __init__(self, model, classIdx, layerName=None):
            
            self.model = model
            self.classIdx = classIdx
            self.layerName = layerName
            
            if self.layerName is None:
                self.layerName = self.find_target_layer()

        def find_target_layer(self):
            
            for layer in reversed(self.model.layers):
            
                if len(layer.output_shape) == 4:
                    return layer.name
            
            raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


        def compute_heatmap(self, image, eps=1e-8):
            
            gradModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layerName).output, self.model.output])

            
            with tf.GradientTape() as tape:
                
                inputs = tf.cast(image, tf.float32)
                (convOutputs, predictions) = gradModel(inputs)
                
                loss = predictions[:, tf.argmax(predictions[0])]
        
            
            grads = tape.gradient(loss, convOutputs)

            
            castConvOutputs = tf.cast(convOutputs > 0, "float32")
            castGrads = tf.cast(grads > 0, "float32")
            guidedGrads = castConvOutputs * castGrads * grads
        
            convOutputs = convOutputs[0]
            guidedGrads = guidedGrads[0]
            
            weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(), (w, h))
            
            numer = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = numer / denom
            heatmap = (heatmap * 255).astype("uint8")
        
            return heatmap

        def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap = cv2.COLORMAP_TWILIGHT):

            heatmap = cv2.applyColorMap(heatmap, colormap)
            heatmap = np.expand_dims(heatmap, axis = 0)
            
            return heatmap
        
    

    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
        
    image = cv2.imread(temp_file_path)

    image = cv2.resize(image,(128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    model = tf.keras.models.load_model("Final_CAM_Model_DL.h5")
    preds = model.predict(image)
    i = np.argmax(preds[0])
    icam = GradCAM(model, i) 
    
    heatmap = icam.compute_heatmap(image)
    image = np.expand_dims(image, axis = 0)
    heatmap = icam.overlay_heatmap(heatmap, image, alpha = 0.5)
    st.markdown("<h3 style='text-align:center; color: white;'>GradCAM Visualization</h3>",unsafe_allow_html=True)
    left_co, cent_co,last_co,l_co,r_co = st.columns(5)
    with last_co:
        st.image(heatmap[0])