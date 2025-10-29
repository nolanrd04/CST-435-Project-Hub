import streamlit as st
from datetime import datetime
import numpy as np
from PIL import Image
import os

# Check if we're running locally or in cloud
IS_LOCAL = os.path.exists("vehicle_classifier_model.keras") or os.environ.get("STREAMLIT_ENV") != "cloud"

# Only import TensorFlow if running locally
if IS_LOCAL:
    try:
        import tensorflow as tf
        from dataProcessor import preprocess_image, IMG_SIZE, CLASS_NAMES
        from CNN import load_data, build_model, train_and_evaluate
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        st.warning("⚠️ TensorFlow not available on the CLOUD. Running in display-only mode.")
else:
    TENSORFLOW_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "vehicle_classifier_model.keras")
accuracy_plot = os.path.join(BASE_DIR, "accuracy_plot.png")
loss_plot = os.path.join(BASE_DIR, "loss_plot.png")

st.set_page_config(page_title="Convolutional Neural Network", page_icon="🧩", layout="centered")
st.title("Convolutional Neural Network - Vehicle Classifier")
st.markdown("The kaggle dataset strictly utilizes images of three vehicles: cars, motorbikes, and airplanes.")
st.markdown("The CNN model is trained to classify these vehicles.")

# Load the trained model only if TensorFlow is available
model = None
if TENSORFLOW_AVAILABLE and os.path.exists(model_path):
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(model_path)
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Tabs
tab1, tab2, tab3 = st.tabs(["Test Model", "Train Model", "Project Requirements"])

# ---------------- Test Model Tab ----------------
with tab1:
    st.title("Test Model")
    st.caption("Upload an image or dataset to test the model.")

    # Upload a single image
    uploaded_img = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        img_array = preprocess_image(uploaded_img)  # already grayscale + normalized
        img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,1)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        st.image(Image.open(uploaded_img), caption=f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.2%})", width='stretch')


# ---------------- Train Model Tab ----------------
with tab2:
    st.title("Training model disabled while deployed. Training happens on the backend.")

# ---------------- Project Requirements Tab ----------------
with tab3:
    st.title("Project Requirements")
    st.markdown("## 1 and 2. Dataset")
    st.markdown('''Our dataset are the Car, Airplane, and Motorbike folders from the Kaggle dataset Natural Images:
    https://www.kaggle.com/datasets/prasunroy/natural-images''')

    st.markdown("## 3. Description")
    st.markdown('''
                Our dataset consists of different images of cars, planes, and motorbikes. These images are on the
                older side, so using images of newer models of cars or planes tends to get incorrect results.
                For example, a 2024 corvette (which arguably looks like a plane) will probably get classified
                as a plane.
                ''')
    st.markdown("## 4. Libraries")
    st.markdown('''
                Tensorflow, Numpy, sklearn, os, matplotlib.

                ## 5-10. Convultion Layers, Pooling, and Flattening
                ''')
    st.code('''
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=input_shape
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            ''')
    st.markdown('''
                ### Max pooling explanation:
                In this code, MaxPooling will move a 2x2 window across the data and take the most important information
                from those windows.

                ## 11. Training and evaluation
                After training and testing our model behind the scenes, it was shown to be 98.59% accurate with a loss of 6.11%.

                ## Results:
                ''')
    st.image(accuracy_plot)
    st.image(loss_plot)

    st.markdown('''
                ## 12. Summarizing the Project
                Our CNN takes images converted to 128x128 black and white np arrays as input, then gets trained on those images
                to classify other vehicles it is given. There are hundreds of images for each vehicle (car, airplane, and motorbike),
                but the images are not super recent, so using new models of cars or airplanes typically yields incorrect results.
                Some ways to improve the model would be to experiment on training with colored images and adding a bunch of recent
                vehicle images to the dataset.
                ''')
    
