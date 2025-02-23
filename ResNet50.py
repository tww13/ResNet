import streamlit as st
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = ResNet101(weights='imagenet')

st.title('Image Classification')
st.write('Upload an image')

uploaded_file = st.file_uploader('Choose an image..', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    st.write('Classifying...')
    
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    decoded_predictions = decode_predictions(prediction, top=3)

    st.image(img, caption='Classified Image', use_container_width=True)

    st.write('Predictions:')
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        st.write(f"{i+1}. {label}: {score:.2f}")
