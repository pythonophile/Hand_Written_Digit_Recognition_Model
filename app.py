import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load MNIST dataset for reference
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load the pre-trained model
model_path = 'model/mnist_model.keras'
model = load_model(model_path)

# Set the page configuration
st.set_page_config(page_title="MNIST Digit Classification", layout="wide")

# Sidebar configuration
st.sidebar.title("MNIST Digit Classification")
num_images = st.sidebar.slider('Number of images to display from test set:', min_value=1, max_value=20, value=5)
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Display app title and description
st.title("🖊️ Handwritten Digit Recognition")
st.markdown("""
    This app classifies handwritten digits from the MNIST dataset. You can either view random images from the test set or upload your own image for prediction!
""")

# Show uploaded image if available
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert the image to grayscale
    image = image.resize((28, 28))  # Resize the image to 28x28

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Preprocess the image for prediction
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28 * 28)  # Reshape to match input shape (1, 784)
    image_array = image_array.astype('float32') / 255  # Normalize pixel values to range [0,1]

    # Make prediction
    pred_prob = model.predict(image_array)
    pred_label = np.argmax(pred_prob, axis=1)[0]

    st.subheader(f"Prediction for Uploaded Image: **{pred_label}**")

# Select random images from test set
sample_indices = np.random.choice(x_test.shape[0], num_images, replace=False)
x_valid = x_test[sample_indices]
y_valid = y_test[sample_indices]

# Reshape the data to fit the model's input expectations
x_valid_reshaped = x_valid.reshape(x_valid.shape[0], 28 * 28)
x_valid_reshaped = x_valid_reshaped.astype('float32') / 255  # Normalize the data

# Predict the probabilities for the selected samples
y_pred_prob = model.predict(x_valid_reshaped)
y_pred = np.argmax(y_pred_prob, axis=1)

# Display the selected images
st.subheader('Selected MNIST Images and Predictions')
fig, axs = plt.subplots(2, num_images, figsize=(15, 4))
for i in range(num_images):
    # Display the image
    axs[0, i].imshow(x_valid[i], cmap='gray')
    axs[0, i].axis('off')
    
    # Display the prediction
    axs[1, i].text(0.5, 0.5, f'Pred: {y_pred[i]}', fontsize=12, ha='center')
    axs[1, i].text(0.5, 0.2, f'True: {y_valid[i]}', fontsize=12, ha='center')
    axs[1, i].axis('off')

st.pyplot(fig)

# Add a footer with developer information
st.markdown("""
    <hr>
    <p style='font-size:44px; color:#4CAF50;'>Developed by <b>Syed Mansoor ul Hassan Bukhari</b></p>
    <p style='font-size:44px;'>For more details, visit the <a href="https://github.com/cyberfantics/handwritten-digit-recognization" style="color:#4CAF50;">GitHub Repository</a>.</p>
""", unsafe_allow_html=True)
