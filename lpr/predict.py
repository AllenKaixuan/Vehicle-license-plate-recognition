import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models


CH_IMAGE_WIDTH = 24
CH_IMAGE_HEIGHT = 48
EN_IMAGE_WIDTH = 20
EN_IMAGE_HEIGHT = 20

CH_LABEL_DICT = {
    'chuan': 0, 'e': 1, 'gan': 2, 'gan1': 3, 'gui': 4, 'gui1': 5, 'hei': 6, 'hu': 7, 'ji': 8, 'jin': 9,
    'jing': 10, 'jl': 11, 'liao': 12, 'lu': 13, 'meng': 14, 'min': 15, 'ning': 16, 'qing': 17, 'qiong': 18, 'shan': 19,
    'su': 20, 'sx': 21, 'wan': 22, 'xiang': 23, 'xin': 24, 'yu': 25, 'yu1': 26, 'yue': 27, 'yun': 28, 'zang': 29,
    'zhe': 30
}
EN_LABEL_DICT = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}
model_ch = models.load_model("models/model_ch.h5")
model_en = models.load_model("models/model_en.h5")


def predict_ch(gray_image):
    # resize, normlize
    resized_image = cv.resize(gray_image, (CH_IMAGE_WIDTH, CH_IMAGE_HEIGHT))
    image = (resized_image - resized_image.mean()) / resized_image.max()
    # predict
    image = tf.reshape(image, (-1, CH_IMAGE_HEIGHT, CH_IMAGE_WIDTH, 1))
    prediction = model_ch.predict(image)

    prediction_index = np.argmax(prediction)
    prediction_results = (
        [k for k, v in CH_LABEL_DICT.items() if v == prediction_index])
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
        [k for k, v in EN_LABEL_DICT.items() if v == prediction_index])
    return prediction_results
