import numpy as np
import cv2 as cv
import tensorflow as tf

from tensorflow.keras import models


CH_IMAGE_WIDTH = 24
CH_IMAGE_HEIGHT = 48
EN_IMAGE_WIDTH = 20
EN_IMAGE_HEIGHT = 20

CH_LABEL = [
    "川", "鄂", "赣", "甘", "贵",
    "桂", "黑", "沪", "冀", "津",
    "京", "吉", "辽", "鲁", "蒙",
    "闽", "宁", "青", "琼", "陕",
    "苏", "晋", "皖", "湘", "新",
    "豫", "渝", "粤", "云", "藏",
    "浙",
]
EN_LABEL = [
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
]

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
    prediction_results = CH_LABEL[prediction_index]
    return prediction_results, prediction[0][prediction_index]


def predict_en(gray_image):
    # resize, normlize
    resized_image = cv.resize(gray_image, (EN_IMAGE_WIDTH, EN_IMAGE_HEIGHT))
    image = (resized_image - resized_image.mean()) / resized_image.max()
    # predict
    image = tf.reshape(image, (-1, EN_IMAGE_HEIGHT, EN_IMAGE_WIDTH, 1))
    prediction = model_en.predict(image)

    prediction_index = np.argmax(prediction)
    prediction_results = EN_LABEL[prediction_index]
    return prediction_results, prediction[0][prediction_index]
