# Real-Time Object Detection using Mobile Robot  Captured Images: A Deep Learning Approach

This repository contains the implementation of deep learning models for object detection and classification tasks. It includes three Jupyter Notebook files, along with an interactive GUI built with Streamlit, and relevant mobile robot images (Hardware).

# Contents
1. Introduction
2. Notebook Files
3. GUI Application
4. Datasets
5. Hardware Images

# Introduction
Deep Learning for Object Detection and Classification is a project aimed at developing and evaluating deep learning models for object detection and classification tasks. The project utilizes convolutional neural networks (CNN) and multilayer perceptron (MLP) classifiers for achieving accurate predictions.

The primary objectives of this project are:
  Implementing a CNN model for object detection
  Building an MLP classifier for object classification
  Evaluating the performance of different regularization techniques
  Developing an interactive GUI application for prediction using the trained models
  
# Notebook Files
1. DLSIP - CNN.ipynb: This Jupyter Notebook file contains the implementation of a convolutional neural network (CNN) model for object detection. 

2. DLSIP - MLP Classifier.ipynb: This Jupyter Notebook file implements a multilayer perceptron (MLP) classifier for object classification. 

3. DLSIP - MLP Regularisation.ipynb: This Jupyter Notebook file demonstrates the implementation of various regularization techniques for improving the performance. It includes techniques such as dropout, L1 and L2 regularization. It provides insights into the impact of regularization on model performance.

# GUI Application
The repository also includes an interactive GUI application built with Streamlit. The application is designed to provide a user-friendly interface for predicting object classes using the trained models. The application can make predictions on both the test dataset and real-time hardware mobile robot captured images.

The following files are associated with the GUI application:

app.py: This Python script contains the code for the Streamlit application. It loads the trained models and provides options for selecting the prediction mode (test dataset or real-time hardware images). It displays the predicted classes based on selected image.

mobilerobot.jpg: This image file is loaded when the Streamlit application is launched. It represents the final mobile robot hardware used in the project.

# Datasets
The primary dataset used for this project is the Open Loris dataset. This dataset consists of a collection of robot-captured images under various lighting conditions. These labeled images are used for training and evaluating the deep learning models implemented in the notebook files.

The Open Loris dataset can be obtained from [https://docs.google.com/document/d/1KlgjTIsMD5QRjmJhLxK4tSHIr0wo9U6XI5PuF8JDJCo/edit].

# Hardware Mobile Robot
The Hardware folder in the repository contains images representing the final mobile robot hardware used in this project. These images obtained from the hardware are used to demonstrate the real-time prediction.
