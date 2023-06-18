#ResNet-34



import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
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

        img = Image.open(img_path).convert("RGB")  # Keep images in RGB format for ResNet-34
        img = img.resize((224, 224))  # Resize image to (224, 224) for ResNet-34

        img = np.asarray(img) / 255.0
        X.append(img)
        y.append(categories.index(category))

# plt.imshow(X[0])  # Display the first preprocessed image


X = np.array(X)
y = np.array(y)

start2 = time.time()

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Residual block function
def residual_block(x, filters, strides=(1, 1), activation='relu'):
    shortcut = x

    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    if strides != (1, 1) or shortcut.shape[3] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)

    return x

# Model Architecture (ResNet-34)
inputs = Input(shape=(224, 224, 3))

x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 64)

x = residual_block(x, 128, strides=(2, 2))
x = residual_block(x, 128)
x = residual_block(x, 128)
x = residual_block(x, 128)

x = residual_block(x, 256, strides=(2, 2))
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)

x = residual_block(x, 512, strides=(2, 2))
x = residual_block(x, 512)
x = residual_block(x, 512)

x = AveragePooling2D((7, 7))(x)
x = Flatten()(x)
x = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

# Model Compilation
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Model Training
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

end2 = time.time()
print(f"Training Time: {(end2 - start2)/60} minutes")

# Save the trained model
model_json = model.to_json()
with open(r"C:\Amrita\Deep Learning\Codes\CNN\model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(r"C:\Amrita\Deep Learning\Codes\CNN\model.h5")

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
plt.title('Resnet34 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training loss and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ResNet34 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


end1 = time.time()
print(f"Total Time: {(end1 - start1)/60} minutes")

