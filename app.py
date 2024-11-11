import json
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import numpy as np
from PIL import Image


def preprocess_image(image):
    """Предобработка изображения для модели"""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_flower(model, image):
    """Предсказание вида цветка"""
    predictions = model.predict(image)
    return predictions


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.extractor = hub.load(url)
    
    def call(self, inputs):
        return self.extractor(inputs)

feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    FeatureExtractorLayer(feature_extractor_url),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(102, activation='softmax')
])

model_path = 'model_weights.weights.h5'
model.load_weights(model_path)
print(model.summary())


st.title("Определение вида цветка по фотографии 🌼")
st.write("Загрузите фотографию, чтобы узнать, какой это цветок.")

with open('label_map.json', 'r') as f:
    class_names = json.load(f)
print(class_names)
uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    st.write("Обработка изображения...")
    processed_image = preprocess_image(image)
    prediction = predict_flower(model, processed_image)
    predicted_class = class_names[str(np.argmax(prediction))]
    st.write(f"Результат: {predicted_class}")