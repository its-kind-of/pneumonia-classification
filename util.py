import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a streamlit app to an image specified by the given image file

    :param image_file:
        The path to the image file to be used as the background
    :return:
        None
    """
    with open(image_file, "rb") as file:
        image_data = file.read()
    b64_encoded = base64.b64encode(image_data).decode()
    style = f"""
        <style>
        .stApp{{
            background-image:url(data:image/base64,{b64_encoded});
            background-size:cover;
            }}
        <style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, model and a list of class names and returns the predicted class and confidence
    score of the image

    :param image : (PIL.Image.Image)
        An image to be classified

    :param model : (tensorflow.keras.Model)
        A trained machine learning model for image classification

    :param class_names : (list)
        A list of class names corresponding to the classes that the model can predict.
    :return:
        A tuple of the predicted class name and the confidence score for that prediction.
    """

    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, round(confidence_score, 2)