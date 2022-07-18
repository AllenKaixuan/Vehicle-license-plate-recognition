import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

TRAIN_DIR = "./train_en"
TEST_DIR = "./enu_train"
MODEL_PATH = "./model/model_en.h5"
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
LABEL_DICT = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


class Cnn(object):
    """
    CNN网络
    """

    def __init__(self):
        pass

    def build_model(self):
        print('build model...')
        self.model = models.Sequential()

        self.model.add(layers.Conv2D(
            6, (5, 5), padding='SAME', activation='sigmoid', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        ))
        self.model.add(layers.AveragePooling2D((2, 2)))

        self.model.add(layers.Conv2D(
            16, (5, 5), padding='SAME', activation='sigmoid'
        ))
        self.model.add(layers.AveragePooling2D((2, 2)))

        self.model.add(layers.Conv2D(
            120, (5, 5), padding='SAME', activation='sigmoid'
        ))
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(84, activation='sigmoid'))
        self.model.add(layers.Dense(CLASSIFICATION_COUNT, activation='sigmoid'))

        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'],
                           )

    # def load_data(self, train_images, train_labels, test_images, test_labels) -> None:
    #     print('载入数据...')
    #     # 归一化
    #     self.train_images = train_images / 255.0
    #     self.test_images = test_images / 255.0
    #
    #     self.train_labels = train_labels
    #     self.test_labels = test_labels

    def load_data(self, dir_path):
        data = []
        labels = []

        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                    resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    data.append(resized_image.ravel())
                    labels.append(LABEL_DICT[item])

        return np.array(data), np.array(labels)

    def onehot_labels(self, labels):
        onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
        for i in np.arange(len(labels)):
            onehots[i, labels[i]] = 1
        return onehots

    def preprocess_data(self):
        train_data, train_labels = self.load_data(TRAIN_DIR)
        test_data, test_labels = self.load_data(TEST_DIR)
        train_data = (train_data - train_data.mean()) / train_data.max()
        test_data = (test_data - test_data.mean()) / test_data.max()
        train_labels = self.onehot_labels(train_labels)
        test_labels = self.onehot_labels(test_labels)
        train_data = tf.reshape(train_data, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        test_data = tf.reshape(test_data, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        self.train_images = train_data
        self.train_labels = train_labels
        self.test_images = test_data
        self.test_labels = test_labels
        # print('train_lables.shape(%s)' % str(self.train_labels.shape))
        print('loading data...')

    def train(self, epoch=50):
        print('training...')
        self.model.fit(self.train_images,
                       self.train_labels,
                       epochs=epoch,
                       shuffle=True
                       )

    def evaluate(self):
        print('evaluating...')
        self.model.evaluate(self.test_images, self.test_labels)

    def save_model(self, model_path=MODEL_PATH):
        print('save model...')
        self.model.save(model_path)

    def load_model(self, model_path=MODEL_PATH):
        print('load model...')
        self.model = models.load_model(model_path)


if __name__ == '__main__':
    cnn = Cnn()
    cnn.preprocess_data()
    cnn.build_model()
    cnn.train()
    cnn.evaluate()
    cnn.save_model()
