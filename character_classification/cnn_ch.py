import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

TRAIN_DIR = "./chs_train"
TEST_DIR = "./chs_test"
MODEL_PATH = "./model/model_ch.h5"
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 48
CLASSIFICATION_COUNT = 31
LABEL_DICT = {
    'chuan': 0, 'e': 1, 'gan': 2, 'gan1': 3, 'gui': 4, 'gui1': 5, 'hei': 6, 'hu': 7, 'ji': 8, 'jin': 9,
    'jing': 10, 'jl': 11, 'liao': 12, 'lu': 13, 'meng': 14, 'min': 15, 'ning': 16, 'qing': 17, 'qiong': 18, 'shan': 19,
    'su': 20, 'sx': 21, 'wan': 22, 'xiang': 23, 'xin': 24, 'yu': 25, 'yu1': 26, 'yue': 27, 'yun': 28, 'zang': 29,
    'zhe': 30
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
            32, (3, 3),
            padding="valid",
            strides=(1, 1),
            data_format="channels_last",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
            activation="relu"
        ))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(
            64, (3, 3), padding="valid",
            strides=(1, 1),
            data_format="channels_last",
            input_shape=(64, 64, 1),
            activation="relu"
        ))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Flatten())
        
        self.model.add(layers.Dense(1024, activation="relu"))
        self.model.add(layers.Dropout(0.4))
        
        self.model.add(layers.Dense(CLASSIFICATION_COUNT, activation="softmax"))

        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'],
                           )

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
                    data.append(resized_image.ravel())  # 展平
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

    def train(self, epoch=30):
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
