import streamlit as st
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

st.title("Класифікація зображень Fashion MNIST")
st.write("Веб-застосунок для порівняння двох моделей: CNN та VGG16")

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

IMG_SIZE_CNN = 28
IMG_SIZE_VGG = 32

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_history(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_for_cnn(image: Image.Image):
    image = image.convert("L")  
    image = image.resize((IMG_SIZE_CNN, IMG_SIZE_CNN))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.expand_dims(img_array, axis=0)    
    return img_array


def preprocess_for_vgg(image: Image.Image):
    image = image.convert("L") 
    image = image.resize((IMG_SIZE_VGG, IMG_SIZE_VGG))
    img_array = np.array(image).astype("float32")

    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.repeat(img_array, 3, axis=-1)     

    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array


def plot_history(history, model_name):
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax_loss.plot(history["val_loss"], label="Val Loss")
    ax_loss.set_title(f"{model_name} - Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    train_acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"

    if train_acc_key in history:
        ax_acc.plot(history[train_acc_key], label="Train Accuracy")
    if val_acc_key in history:
        ax_acc.plot(history[val_acc_key], label="Val Accuracy")

    ax_acc.set_title(f"{model_name} - Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    return fig_loss, fig_acc

st.sidebar.header("Налаштування")

selected_model = st.sidebar.selectbox(
    "Оберіть модель",
    ["CNN", "VGG16"]
)

if selected_model == "CNN":
    model_path = "cnn_model.keras"
    history_path = "cnn_history.json"
    model_name = "CNN"
else:
    model_path = "vgg16_model.keras"
    history_path = "vgg16_history.json"
    model_name = "VGG16"

model = load_model(model_path)
history = load_history(history_path)

st.subheader(f"Графіки навчання моделі: {model_name}")

fig_loss, fig_acc = plot_history(history, model_name)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_loss)
with col2:
    st.pyplot(fig_acc)

st.subheader("Завантаження зображення")

uploaded_file = st.file_uploader(
    "Оберіть зображення для класифікації",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_result = st.columns([1, 1.2])

    with col_img:
        st.image(image, caption="Завантажене зображення", use_container_width=True)

    if selected_model == "CNN":
        processed_image = preprocess_for_cnn(image)
    else:
        processed_image = preprocess_for_vgg(image)

    predictions = model.predict(processed_image)
    probabilities = predictions[0]
    predicted_class_idx = int(np.argmax(probabilities))
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    predicted_confidence = float(np.max(probabilities))

    with col_result:
        st.subheader("Результат класифікації")
        st.success(
            f"Передбачений клас: **{predicted_class_name}** "
            f"({predicted_confidence * 100:.2f}%)"
        )

        st.write("### Ймовірності для кожного класу")
        probs_dict = {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        }

        st.dataframe(
            {
                "Клас": list(probs_dict.keys()),
                "Ймовірність": [f"{v * 100:.2f}%" for v in probs_dict.values()]
            },
            use_container_width=True
        )

        st.write("### Графік ймовірностей")
        st.bar_chart(probs_dict)