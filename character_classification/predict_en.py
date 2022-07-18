import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from character_classification import cnn_en

IMAGE_PATH = './'
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
MODEL_PATH = "./character_classification/model/model_en.h5"


class Predict:
    def load_data(self):
        gray_image = cv.imread(IMAGE_PATH,cv.IMREAD_GRAYSCALE)
        resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = (resized_image - resized_image.mean()) / resized_image.max()
        image = tf.reshape(image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))  # tf.reshape参数不能为None
        self.image = image

    def load_model(self, model_path=MODEL_PATH):
        print('load model...')
        self.model = models.load_model(model_path)

    def predict(self):
        self.load_data()
        # self.load_model()
        prediction = self.model.predict(self.image)
        prediction_index = np.argmax(prediction)
        prediction_results = ([k for k, v in cnn_en.LABEL_DICT.items() if v == prediction_index])
        return  prediction_results
