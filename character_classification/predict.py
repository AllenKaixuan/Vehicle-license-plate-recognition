import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from character_classification import cnn_ch
from character_classification import cnn_en


CH_IMAGE_WIDTH = 24
CH_IMAGE_HEIGHT = 48
EN_IMAGE_WIDTH = 20
EN_IMAGE_HEIGHT = 20

model_ch = models.load_model("character_classification/model/model_ch.h5")
model_en = models.load_model("character_classification/model/model_en.h5")


def predict_ch(gray_image):
    # resize, normlize
    resized_image = cv.resize(gray_image, (CH_IMAGE_WIDTH, CH_IMAGE_HEIGHT))
    image = (resized_image - resized_image.mean()) / resized_image.max()
    # predict
    image = tf.reshape(image, (-1, CH_IMAGE_HEIGHT, CH_IMAGE_WIDTH, 1))
    prediction = model_ch.predict(image)

    prediction_index = np.argmax(prediction)
    prediction_results = (
        [k for k, v in cnn_ch.LABEL_DICT.items() if v == prediction_index])
    return prediction_results


def predict_en(gray_image):
    # resize, normlize
    resized_image = cv.resize(gray_image, (EN_IMAGE_WIDTH, EN_IMAGE_HEIGHT))
    image = (resized_image - resized_image.mean()) / resized_image.max()
    # predict
    image = tf.reshape(image, (-1, EN_IMAGE_HEIGHT, EN_IMAGE_WIDTH, 1))
    prediction = model_en.predict(image)

    prediction_index = np.argmax(prediction)
    prediction_results = (
        [k for k, v in cnn_en.LABEL_DICT.items() if v == prediction_index])
    return prediction_results
