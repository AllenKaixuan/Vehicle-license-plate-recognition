from Utilities.io import DataLoader
from Utilities.lossMetric import *
from Utilities.trainVal import MinMaxGame
from GAN.RRDBNet import RRDBNet
from GAN.Disciminator import Discriminator
import tesorflow.keras as keras
import numpy as np
import glob #glob支持通配符查找 

PATH = 'PATH_TO_OUTPUT_DIR/192_96' 
SAVE_PATH = 'Models/'

files = glob.glob(PATH + '/*.jpg') * 3  # 数据扩张，相同图片有着不同的对比度
np.random.shuffle(files)
train, val = files[:int(len(files)*0.8)], files[int(len(files)*0.8):]
loader = DataLoader()
trainData = DataLoader().load(train, batchSize=16)
valData = DataLoader().load(val, batchSize=64)


discriminator = Discriminator()
extractor = buildExtractor()
generator = RRDBNet(blockNum=10)


# 在 SRGAN 论文中定义的损失函数
def contentLoss(y_true, y_pred):
    featurePred = extractor(y_pred)
    feature = extractor(y_true)
    mae = tf.reduce_mean(keras.losses.mae(y_true, y_pred))
    return 0.1*tf.reduce_mean(keras.losses.mse(featurePred, feature)) + mae

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
generator.compile(loss=contentLoss, optimizer=optimizer, metrics=[psnr, ssim])

history = generator.fit(x=trainData, validation_data=valData, epochs=1, steps_per_epoch=300, validation_steps=100)



PARAMS = dict(lrGenerator = 1e-4, 
              lrDiscriminator = 1e-4,
              epochs = 1, 
              stepsPerEpoch = 500, 
              valSteps = 100)
game = MinMaxGame(generator, discriminator, extractor)
log, valLog = game.train(trainData, valData, PARAMS)
# ideally peak signal noise ratio(snRation or psnr) should reach ~22

