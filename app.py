from model_prediction import predict
import streamlit as st
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center;'>Sign Language Detection</h1>",
            unsafe_allow_html=True)

img_file = st.sidebar.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    try:
        image = Image.open(BytesIO(img_file.read()))
        btn = st.sidebar.button("Get Landmarks")
        if btn:
            landmark_img, pred = predict(image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_column_width=True, caption="Original")
            with col2:
                st.image(landmark_img, use_column_width=True,
                         caption="Predicted")
            st.markdown(f"<h3 style='text-align: center;'>Model Predicted : {pred.upper()}</h3>",
                        unsafe_allow_html=True)

    except Exception as e:
        st.write(e)
        st.write("Unable to open file")
