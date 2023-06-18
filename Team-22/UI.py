import streamlit as st
import cv2
import os
import time
from PIL import Image
from timeit import default_timer as timer
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import tensorflow as tf

# Import your style transfer code here

RESIZE_HEIGHT = 607

start_time = timer()

# Weights of the different loss components
CONTENT_WEIGHT = 8e-4  # 8e-4
STYLE_WEIGHT = 8e-1  # 8e-4

STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

st.write("Team-22")
st.write("The Team Members are Ashwaj ,Tanvi ,Hitesh in this Project")

# Set up Streamlit app
st.title("Image Style Transfer")

CONTENT_LAYER_NAME = "block5_conv2"

# Upload content and style images
content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

def style_loss(style_features, combination_features, combination_size):
    S = gram_matrix(style_features)
    C = gram_matrix(combination_features)
    channels = style_features.shape[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def deprocess_image(tensor, result_height, result_width):
    tensor = tensor.numpy()
    tensor = tensor.reshape((result_height, result_width, 3))

    # Remove zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680

    # 'BGR'->'RGB'
    tensor = tensor[:, :, ::-1]
    return np.clip(tensor, 0, 255).astype("uint8")


def get_optimizer():
    return keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=8.0, decay_steps=445, decay_rate=0.98
            # initial_learning_rate = 2.0, decay_steps = 376, decay_rate = 0.98
        )
    )

def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    loss_content = compute_content_loss(content_features, combination_features)
    loss_style = compute_style_loss(style_features, combination_features,
                                    combination_image.shape[1] * combination_image.shape[2])

    return CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

def compute_content_loss(content_features, combination_features):
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]

    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2


def get_model():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    base_model = vgg19.VGG19(weights='imagenet', include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in base_model.layers])

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
    return keras.Model(inputs=base_model.inputs, outputs=outputs_dict)


def preprocess_image(image_path, target_height, target_width):
    img = keras.preprocessing.image.load_img(image_path, target_size=(target_height, target_width))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = vgg19.preprocess_input(arr)
    return tf.convert_to_tensor(arr)

def compute_style_loss(style_features, combination_features, combination_size):
    loss_style = 0

    for layer_name in STYLE_LAYER_NAMES:
        style_feature = style_features[layer_name][0]
        combination_feature = combination_features[layer_name][0]
        loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)

    return loss_style

# Rest of the code...

if content_image and style_image:
    # Load content and style images
    content_img = Image.open(content_image)
    style_img = Image.open(style_image)

    # Display content and style images
    st.subheader("Content Image")
    st.image(content_img, caption="Content Image", use_column_width=True)

    st.subheader("Style Image")
    st.image(style_img, caption="Style Image", use_column_width=True)

    # Slider for NUM_ITER
    num_iter = st.number_input("Number of Iterations", min_value=1, max_value=20000, step=10)

    # Perform style transfer
    if st.button("Transfer Style"):
        # Convert images to arrays
        content_array = np.array(content_img)
        style_array = np.array(style_img)

        # Preprocess images
        content_path = "content.jpg"
        style_path = "style.jpg"
        content_img.save(content_path)
        style_img.save(style_path)
        content_tensor = preprocess_image(content_path, RESIZE_HEIGHT, content_img.size[1])
        style_tensor = preprocess_image(style_path, RESIZE_HEIGHT, style_img.size[1])

        # Resize the generated image to match the content image
        generated_image = tf.Variable(
            tf.random.uniform(content_tensor.shape, dtype=tf.dtypes.float32)
        )

        # Rest of the code...
        NUM_ITER = num_iter  # Update NUM_ITER value
        # Build model
        model = get_model()
        optimizer = get_optimizer()

        content_features = model.predict(content_tensor)
        style_features = model(style_tensor)

        # Optimize result image
        for iter in range(NUM_ITER):
            with tf.GradientTape() as tape:
                loss = compute_loss(model, generated_image, content_features, style_features)

            grads = tape.gradient(loss, generated_image)
            optimizer.apply_gradients([(grads, generated_image)])

            st.text(f"Iteration: {iter + 1}/{NUM_ITER}")
            st.text(f"Weight: {loss.numpy()}")

        # Generate stylized image
        stylized_image = deprocess_image(generated_image, RESIZE_HEIGHT, generated_image.shape[1])

        # Display stylized image
        st.subheader("Stylized Image")
        st.image(stylized_image, caption="Stylized Image", use_column_width=True)

        # Save stylized image
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "stylized_image.jpg")
        Image.fromarray(stylized_image).save(output_path)

        end_time = time.time()
        total_time = end_time - start_time
        st.text(f"Total Time Taken: {total_time} seconds")