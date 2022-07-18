import os
import cv2 as cv
import numpy as np
import numpy as np
from plate_location import hsv_plate_locator
from character_segmentation import segmentation
from character_classification import predict_ch, predict_en

IMAGE_PATH = "./input/"
CHARACTER_PATH = "./output/character_output/"


def load_data():
    image_path = []
    for item in os.listdir(IMAGE_PATH):  # 实际只读入一张图片
        image_path.append(item)
    print('loading data...')
    return image_path


def plate_locator(image_path):
    for item in image_path:
        hsv_plate_locator.IMAGE_PATH = './input/' + item  # 慎用相对路径,是针对当前文件.注意分隔符的有无
        hsv_plate_locator.SAVE_PATH = './output/licence_output/'
        hsv_plate_locator.locate()
    print('locating...')


def main_segmentation():
    segmentation.IMAGE_PATH = './output/licence_output/'
    segmentation.SAVE_PATH = './output/character_output/'
    seg = segmentation.Segmentation()

    seg.segment()
    print('segmenting...')


def classification():
    output = ''
    predict_ch.IMAGE_PATH = os.path.join(CHARACTER_PATH, '0.jpg')
    predict_ch.MODEL_PATH = "./character_classification/model/model_ch.h5"
    predict_en.MODEL_PATH = "./character_classification/model/model_en.h5"
    prediction_ch = predict_ch.Predict()
    prediction_en = predict_en.Predict()
    prediction_ch.load_model(predict_ch.MODEL_PATH)
    prediction_en.load_model(predict_en.MODEL_PATH)

    output += str(prediction_ch.predict())
    for image in range(1, 7):
        predict_en.IMAGE_PATH = os.path.join(CHARACTER_PATH, str(image) + '.jpg')
        output += str(prediction_en.predict())
    print('result:', output)


if __name__ == "__main__":
    data = load_data()
    plate_locator(data)
    main_segmentation()
    classification()
