import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

data_dir = r"D:\Amrita\6th sem\Deep Learning for Signal & Image Processing\Dataset"
categories = ["good", "bad"]

X = []
y = []
for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        X.append(img)
        y.append(categories.index(category))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import time

# CNN Model Architecture Definition
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64,  3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Model Compilation
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
 
# Model Training
start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
 
# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss: ", loss)
print("Test Accuracy: ", accuracy)



 


# Test the network with your test set and observe the metric
predictions = model.predict(X_test, batch_size=32)
model_train_compile_predict_time = time.time() - start_time
print(classification_report(y_test, predictions.round()))
print("Model Train, Compile and Predict Time: ", model_train_compile_predict_time)

# Make a plot of training loss and validation loss to check for the regular fit of the trained network
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Make a plot of training accuracy and validation accuracy to check for the regular fit of the trained network
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


print("Test Loss: ", loss)
print("Test Accuracy: ", accuracy)
print("Train Loss: ", history.history['loss'][-1])
print("Train Accuracy: ", history.history['accuracy'][-1])
print("Validation Loss: ", history.history['val_loss'][-1])
print("Validation Accuracy: ", history.history['val_accuracy'][-1])
# Calculate validation loss and accuracy
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print("Validation Loss: ", val_loss)
print("Validation Accuracy: ", val_accuracy)





# Assuming you have a sample image path
sample_image_path = r"D:\Amrita\6th sem\Deep Learning for Signal & Image Processing\Dataset\good\82.jpg"

# Read and preprocess the sample image
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.resize(sample_image, (64, 64))
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
sample_image = sample_image.astype("float32") / 255.0

# Reshape the image to match the input shape of the model
sample_image = np.expand_dims(sample_image, axis=0)

# Predict the category of the sample image
prediction = model.predict(sample_image)

# Print the predicted category
if prediction[0] > 0.5:
    print("Prediction: bad")
else:
    print("Prediction: good")
