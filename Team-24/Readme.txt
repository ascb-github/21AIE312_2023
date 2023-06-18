1.dataset folder should be loaded into the dataset_creation.ipynb from that faces folder will be generated ant that folder should be given for genrating embedding .

2.In final code give the csv file which have the image embedding values of all images(CSV file will be generated from step 1)

3.Install Packages: Install the required packages using the pip install command. The required packages are keras_vggface, keras_applications, PyYAML.

4.Import Libraries: Import the necessary libraries in your code. These include PIL, cv2, os, scipy, tensorflow, keras, csv, pandas, and matplotlib.pyplot.

5.Load the Dataset: Use the read_csv function to load your dataset from the specified file path. Make sure the file exists and contains the required columns.

6.Split the Dataset: Split your dataset into training, validation, and test sets using the train_test_split function from sklearn.model_selection.

7.Define and Compile the Model: Create your fully connected model using the Sequential class from keras.models. Define the layers and compile the model with the desired optimizer, loss function, and metrics.

8.Train the Model: Fit the model on the training data using the fit function. Specify the batch size, number of epochs, and validation data.

9.Evaluate the Model: Evaluate the model's performance on the test set using the evaluate function.

10.Set up Face Detection: Import the necessary libraries for face detection, such as cv2, and define the functions required for face detection, image embedding, triplet loss, and removing files in a directory.

11.Set up the Gradio Interface: Install the gradio library using pip install gradio. Create a Gradio interface to take inputs of reference and group images and output the predicted label.
