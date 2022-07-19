import os
from unittest import result
import cv2 as cv
import numpy as np
import numpy as np
from plate_location import hsv_plate_locator
from character_segmentation import segmentation
from character_classification import predict

IMAGE_PATH = "./input/"
CHARACTER_PATH = "./output/character_output/"


def recognize(image):
    # locate
    plates_image = hsv_plate_locator.locate(image)
    # segment
    seg = segmentation.Segmentation()
    chars_image = seg.segment(plates_image)
    result = []
    for i in range(len(chars_image)):
        output = []
        output += predict.predict_ch(chars_image[i][0])
        for j in range(1, len(chars_image[i])):
            output += predict.predict_en(chars_image[i][j])
        result += output

    return result


if __name__ == "__main__":
    image_path = './input/img_3.png'
    image = cv.imread(image_path)
    recognize(image)
