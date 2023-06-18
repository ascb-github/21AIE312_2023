Amrita ID Detection and Text Detection
This project focuses on two tasks: Amrita ID detection and text detection followed by OCR. It provides the necessary code and datasets to perform these tasks. The project includes the following files:
1.	Amrita ID DETECTION data.zip: This compressed file contains the dataset for the Amrita ID detection task. It includes images people wearing Amrita ID cards.
2.	TEXT DETECTION data.zip: This compressed file contains the dataset of Amrita ID cards for the text detection task. It includes images with ID cards in various scenarios.
3.	Yolov5.ipynb: This Jupyter Notebook file contains the code for implementing the YOLOv5 model for both the Amrita ID detection and text detection tasks. The notebook provides instructions on how to run the code and train the model using the respective datasets.
4.	Yolov8.ipynb: This Jupyter Notebook file contains an alternative implementation of the YOLOv8 model for the same tasks. It can be used as an alternative approach to compare results or choose between the two models. The notebook also includes instructions on how to run the code and train the model using the provided datasets.
5.	InformatonExtractionUI.py: This Python file contains the code for the user interface (UI) related to the information extraction task. It allows users to input an image of a person wearing an Amrita ID card or a person without it and extracts relevant information using OCR from the detected Amrita ID card using the trained model.
6.	TextDetectionUI.py: This Python file contains the code for the user interface (UI) related to the information extraction task. It allows users to input an image of a person wearing an Amrita ID card or a person without it and detects the text in the ID Card and performs OCR on it.
Please note that the necessary dependencies and packages should be installed before running the code files. Ensure that the dataset paths are correctly set within each model file. 
Roboflow in necessary to import datasets and models in the form of an API Key.
pip install roboflow
To run the UI codes, make sure that Streamlit is installed along with other required packages. Execute the command streamlit run InformatonExtractionUI.py for the information extraction UI and streamlit run TextDetectionUI.py for the text detection UI.
Team Members:
•	Jaswanth Kunisetty - BL.EN.U4AIE20031
•	Pranav Ramachandrula - BL.EN.U4AIE20051
•	Sruthi S - BL.EN.U4AIE20055
Feel free to reach out to us if you have any questions or require further assistance with the project.
