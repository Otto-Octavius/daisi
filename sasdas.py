import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, optimizers, callbacks


def st_ui():
    st.set_page_config(layout="wide")
    st.title("Compute edges")
    user_image = st.sidebar.file_uploader("Load your own image")
    im = Image.open('547.png')
    st.header("Original image")
    st.image(im)


if __name__ == "__main__":
    st_ui()
