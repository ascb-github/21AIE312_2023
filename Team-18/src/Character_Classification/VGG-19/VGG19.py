#VGG - 19

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time

start1 = time.time()

data_dir = '/content/drive/MyDrive/Colab Notebooks/DATA_LABELS_FINAL/'
categories = os.listdir(data_dir)

X = []
y = []
for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        img = Image.open(img_path).convert("RGB")  # Keep images in RGB format for VGG-19
        img = img.resize((224, 224))  # Resize image to (224, 224) for VGG-19

        img = np.asarray(img) / 255.0
        X.append(img)
        y.append(categories.index(category))

# plt.imshow(X[0])  # Display the first preprocessed image


X = np.array(X)
y = np.array(y)

start2 = time.time()

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Architecture (VGG-19)
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

# Model Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Model Training
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

end2 = time.time()
print(f"Training Time: {(end2 - start2)/60} minutes")

# Save the trained model
# model.save("trained_model.h5")

model_json = model.to_json()
with open('/content/drive/MyDrive/Colab Notebooks/h5files/VGG19_model.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights('/content/drive/MyDrive/Colab Notebooks/h5files/VGG19_model.h5')

# Extract accuracy and loss values from history
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Print the last values of accuracy and loss during training
print("Training Accuracy:", train_accuracy[-1])
print("Training Loss:", train_loss[-1])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Testing Accuracy:", test_accuracy)
print("Testing Loss:", test_loss)

# Print the last values of accuracy and loss during validation
print("Validation Accuracy:", val_accuracy[-1])
print("Validation Loss:", val_loss[-1])

# Classification report
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test, np.argmax(predictions, axis=1), target_names=categories))

#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Plotting training accuracy and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training loss and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

end1 = time.time()
print(f"Total Time: {(end1 - start1)/60} minutes")
