# XAI in Alzheimer's detection

This repository contains the code, dataset, and UI files for the classification of Alzheimer's disease using brain MRI images. The project consists of three main folders: `ALzheimersDataset`, `Code`, and `UI`.Additionally, the model interpreted using the XAI (Explainable Artificial Intelligence) technique called GradCAM.

### ALzheimersDataset

The `ALzheimersDataset` folder contains the dataset used for training and testing the model. It is organized into two subfolders:

- `train`: This folder contains the training images divided into four classes:
  - `MildDemented`
  - `ModerateDemented`
  - `NonDemented`
  - `VeryMildDemented`

- `test`: This folder contains the test images divided into the same four classes as in the training set.
The dataset can be downloaded from : https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

### Code

The `Code` folder contains the implementation of the model for Alzheimer's disease classification. The code is written in Python using the Keras library.We have developed a CNN architecture that can classify the stages of Alzheimer's disease. 

### UI

The `UI` folder contains files related to the user interface for interacting with the trained model. The files included are:

- `Final_Model_DL.h5`: This file contains the trained model weights.
- `GUI.py`: This file implements a graphical user interface using a Python GUI framework.
- `img.png`: An image file used in the GUI.
- `mona1.png`: Another image file used in the GUI.

### Usage

To use this repository, you can follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the necessary dependencies installed (Python, Keras, etc.).
3. Download the `ALzheimersDataset` from kaggle, or use the provided dataset.
4. Train the model using the provided code in the `Code` folder.
5. Once trained, you can use the UI files in the `UI` folder to interact with the model and perform classification on new images.


### Team 04
Savarala Chethana - BL.EN.U4AIE20059
Sreevathsa Sree Charan - BL.EN.U4AIE20062
Vemula Srihitha - BL.EN.U4AIE20072