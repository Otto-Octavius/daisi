import PIL
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_dce_net(image_size=None) -> keras.Model:
    '''
    This function builds the DCE network since there is no currect option to save the model as a pickle, h5, YAML format for a Keras subclass 
    model. But luckily, the weights can be loaded. Thus, we use the declared model and set the weights for our use. 
    :param int image_size: Set to none since no training is done. This function will be called in an Keras Object 
    :return keras.Model: A Keras model is returned with it's Conv layers performing Deep Curve Estimation, 
    '''

    input_image = keras.Input(shape=[image_size, image_size, 3])
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_image)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
    return keras.Model(inputs=input_image, outputs=x_r)


class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = build_dce_net()

    def get_enhanced_image(self, data, output):
        x = data
        for i in range(0, 3 * 8, 3):
            r = output[:, :, :, i: i + 3]
            x = x + r * (tf.square(x) - x)
        return x

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def test_step(self, data):
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )


def get_model():
    '''
    This void function just returns the Subclass model with updated weights.  
    '''
    return ZeroDCE()


model = get_model()
model.load_weights("optimumW.h5")


def enhance(i):
    '''
    This function is where the translation occurs, the enhancement function takes in the PIL image and converts it to an array.t 
    :return output_image: Enhanced Image
    '''
    image = keras.preprocessing.image.img_to_array(i)
    image = image[:, :, :3] if image.shape[-1] > 3 else image
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image


def st_ui():
    '''
    TThis Streamlit UI keeps running in an indefinite basis through __main__. It takes in the an Image or has a default demo image.
    It also has a button to bring out the details that are hid in the dark. The output image is the taken from the enhance() function
    and displayed.
    '''
    st.title("Image Enhancement in Low Light Conditions :flashlight:")
    user_image = st.sidebar.file_uploader("Load your own image")
    if user_image is not None:
        i = Image.open(user_image)
    else:
        i = Image.open('547.png')
    w, h = i.size
    if h > 720:
        i = i.resize((int((float(i.size[0]) * float((720 / float(i.size[1]))))), 720), PIL.Image.NEAREST)
    st.header("Original image")
    st.image(i)
    draw_landmark_button = st.button('Bring out Details')
    result = enhance(i)
    if draw_landmark_button:
        st.header("Enhanced Image")
        st.image(result)


if __name__ == '__main__':
    st_ui()
