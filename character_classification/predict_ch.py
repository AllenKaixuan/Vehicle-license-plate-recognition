import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import cnn_ch

IMAGE = './meng.jpg'
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 48




class Predict:
    def load_data(self):
        image = cv.imread(IMAGE)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print('gray image shape:', gray_image.shape)
        resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = (resized_image - resized_image.mean()) / resized_image.max()
        image = tf.reshape(image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))  # tf.reshape参数不能为None
        self.image = image

    def load_model(self, model_path=cnn_ch.MODEL_PATH):
        print('load model...')
        self.model = models.load_model(model_path)

    def predict(self):
        prediction = self.model.predict(self.image)
        prediction_index = np.argmax(prediction)
        prediction_results = ([k for k, v in cnn_ch.LABEL_DICT.items() if v == prediction_index])
        print('predict results:', prediction_results)


if __name__ == '__main__':
    Prediction = Predict()
    Prediction.load_data()
    Prediction.load_model()
    Prediction.predict()
