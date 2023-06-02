import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify
# title
st.title("Pneumonia classification")

# header
st.header("Please upload an image of a chest X-ray")

# uploading file
uploaded_file = st.file_uploader("", type=['jpeg', 'png', 'jpg'])

# load classifier
model = load_model(r"model/pneumonia_classifier.h5")

# load class name
with open(r"model/labels.txt", "r") as file:
    class_names = [line[:-1].split(" ")[1] for line in file.readlines()]
    file.close()


# display
if uploaded_file is not None:
	image = Image.open(uploaded_file).convert("RGB")
	st.image(image, use_column_width=True)

	# classify image
	class_name, confidence_score = classify(image, model, class_names)

	# write classification
	st.write(f"## {class_name}")
	st.write(f"## score : {confidence_score}")
