from Utilities.io import DataLoader
from GAN.RRDBNet import RRDBNet
import tensorflow as tf
import glob
import numpy as np
import cv2 as cv

DATA_PATH = 'Samples'
SAVE_PATH = 'Output/'
MODEL_PATH = 'Models/rrdb'


def generate():
    loader = DataLoader()
    data = loader.load(glob.glob(DATA_PATH + '/*.jpg'), batchSize=1)
    model = RRDBNet(blockNum=10)
    model.load_weights(MODEL_PATH)
    for downSample, original in data.take(1):  # 只输入一张图片
        Pred = model.predict(downSample)
        Pred = np.squeeze(np.clip(Pred, a_min=0, a_max=1))
        Pred = cv.cvtColor(Pred, cv.COLOR_BGR2RGB)
        cv.imwrite(SAVE_PATH + '.jpg', Pred * 255)


if __name__ == "__main__":
    generate()
