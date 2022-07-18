import cv2 as cv
import numpy as np
import os

IMAGE_HEIGHT = 105
IMAGE_WIDTH = 400
IMAGE_PATH = "./image"
SAVE_PATH = "./result/"

class Segmentation():
    def __init__(self):
        pass

    # 加载图像数据集和标签，其中标签是通过拆解中科大车牌数据集的图像名称得到的
    def load(self, path, width, height):
        input_data = []
        # 导入指定路径下的所有图像数据
        for item in os.listdir(path):
            imgpath = os.path.join(path, item)
            # filename = copy.copy(item)
            if os.path.isdir(path):
                input_image_ini = cv.imread(imgpath)
                input_image = cv.resize(input_image_ini, (width, height))
                input_data.append(input_image)

        # 返回读取的图像数据集dataset和得到的对应图像的标签集label
        return input_data

    # 数据探查，可以显示刚才导入的图像数据集
    def image_data_show(self, input_image):
        for img in input_image:
            cv.imshow("image_show", img)
            cv.waitKey()
        cv.destroyAllWindows()

    # 在进行车牌字符分割前，先对输入图像的尺寸进行调整
    def normal_shape(self, input_image, width, height):
        output_image = cv.resize(input_image, (width, height))
        return output_image

    # 将输入的车牌图像进行灰度化和二值化操作
    def Binarization(self, input_image):
        # 灰度化和二值化该区域
        grey_image_ini = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        # grey_image = cv.GaussianBlur(grey_image_ini, (3, 3), 0)
        # 使用双边滤波方法去除噪点
        grey_image = cv.bilateralFilter(grey_image_ini, 11, 17, 17)
        ret, bin_img = cv.threshold(grey_image, 0, 255, cv.THRESH_OTSU)

        # 去除边框
        offsetX = 3
        offsetY = 5
        offset = bin_img[offsetY:-offsetY, offsetX:-offsetX]
        working_region = np.copy(offset)
        offset_region = np.copy(offset)

        return working_region, offset_region

    # 对二值化图像中的汉字进行高斯模糊，使其在字符分割时可以被看作一个整体
    def Blur(self, input_image):
        hanzi_max_width = input_image.shape[1] // 8;  # 假设汉字最大宽度为整个车牌宽度的1/8
        hanzi_region = input_image[:, 0:hanzi_max_width]
        cv.GaussianBlur(hanzi_region, (9, 9), 0, dst=hanzi_region)

    # 寻找波峰，用于去除车牌的上下边框
    def find_waves(self, threshold, histogram):
        up_point = -1
        is_peak = False
        if histogram[0] > threshold:
            up_point = 0
            is_peak = True
        wave_peaks = []
        for i, x in enumerate(histogram):
            if is_peak and x < threshold:
                if i - up_point > 2:
                    is_peak = False
                    wave_peaks.append((up_point, i))
            elif not is_peak and x >= threshold:
                is_peak = True
                up_point = i
        if is_peak and up_point != -1 and i - up_point > 4:
            wave_peaks.append((up_point, i))

        return wave_peaks

    # 使用二值化图像的黑白波峰值，去除车牌的上下边框
    def remove_border(self, input_image):
        # 按行求出每列的像素点值的和，并通过和阈值比较，找到图像中的波峰
        row_histogram = np.sum(input_image, axis=1)
        row_min = np.min(row_histogram)
        row_average = np.sum(row_histogram) / input_image.shape[0]
        row_threshold = (row_min + row_average) / 2
        wave_peaks = self.find_waves(row_threshold, row_histogram)

        # 通过水平方向上波（白点，即二值化后图像上的字符）的宽度，找到车牌字符所在的具体范围，并去除掉周围黑色背景中的上下边框
        # 由于此程序使用了车牌字符的比例大小作为字符分割的重要依据，所以必须去掉车牌图片中的多余背景，提高字符分割的准确率
        wave_span = 0.0
        for wave_peak in wave_peaks:
            span = wave_peak[1] - wave_peak[0]
            if span > wave_span:
                wave_span = span
                selected_wave = wave_peak

        completed_image = input_image[selected_wave[0]:selected_wave[1], :]

        return completed_image

    # 字符拆分函数，结果返回拆分后得到的单个字符图像列表
    def char_split(self, binary_input_image, offset_input_image):
        # 使用findContours函数，来识别车牌图像上的字符轮廓
        # 使用了RETR_EXTERNAL方法来检测字符轮廓，其只检测图形的外轮廓；使用了CHAIN_APPROX_SIMPLE的轮廓近似方法，只通过图形的边缘点来近似图形轮廓
        char_contours, _ = cv.findContours(binary_input_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 设置车牌字符的大小范围
        min_width = binary_input_image.shape[1] // 40
        min_height = binary_input_image.shape[0] * 7 // 10
        valid_char_regions = []
        # 开始对检测到的字符图形按条件进行筛选
        for i in np.arange(len(char_contours)):
            x, y, w, h = cv.boundingRect(char_contours[i])
            # 按照上面得到的字符高度和宽度进行条件筛选
            if h >= min_height and w >= min_width:
                valid_char_regions.append((x, offset_input_image[y:y + h, x:x + w]))
        # 将按照车牌上的x坐标从左到右排序
        sorted_regions = sorted(valid_char_regions, key=lambda region: region[0])

        # 将上面分割得到的字符，放到一个list中返回
        candidate_char_images = []
        for i in np.arange(len(sorted_regions)):
            candidate_char_images.append(sorted_regions[i][1])

        return candidate_char_images

    # 二值化图像显示函数，用于图像二值化后的数据探查
    def binary_show(self, input_data):
        for i in range((len(input_data))):
            reshape_img = self.normal_shape(input_data[i], IMAGE_WIDTH, IMAGE_HEIGHT)
            b_temp_img, off_temp_img = self.Binarization(reshape_img)
            self.Blur(b_temp_img)
            b_img = self.remove_border(b_temp_img)
            b_img = cv.resize(b_img, (300, 80))
            cv.imshow("binary_show", b_img)
            cv.waitKey()
        cv.destroyAllWindows()

    # 字符分割函数，调用前面声明的几个方法，进行字符分割
    def character_segmentation(self, input_data):
        output_img = []

        for i in range(len(input_data)):
            # 依次调用几个方法，对车牌图像进行字符分割
            reshape_img = self.normal_shape(input_data[i], 308, 83)
            b_temp_img, off_temp_img = self.Binarization(reshape_img)
            self.Blur(b_temp_img)
            b_img = self.remove_border(b_temp_img)
            off_img = self.remove_border(off_temp_img)
            output_img.append(self.char_split(b_img, off_img))
        # output_img = np.array(output_img, dtype=np.uint8)


        return output_img

    # 字符分割结果展示，显示分割得到的所有单个字符图像
    def result_show(self, candidate_chars):

        for i in range(len(candidate_chars)):
            for char in candidate_chars[i]:
                image = cv.resize(char, (50, 50))

    # 将所有得到的单个字符图像保存到result文件夹下，并在文件名中记录其标签值
    def image_save(self, input_data):

        tosaves = np.array(input_data)
        count = np.arange(0, (tosaves.shape[0] + 1) * tosaves.shape[1], 1)
        j = -1
        for i in range(len(input_data)):
            for tosave in tosaves[i]:
                j = j + 1
                # 将所有得到的单个字符图像统一调整为20*20大小
                img = cv.resize(tosave, (20, 20))
                # 将所有得到的图像全部保存在result目录下，统一保存为20*20大小的jpg格式文件
                # 将所有得到的单个字符图像命名为：“00x_0y_label”的格式，其中的x为当前图像所对应的输入图像的编号，
                # y为当前图像所显示字符在输入图像中的位置，label则对应了该图像的标签值
                cv.imwrite(SAVE_PATH+str(count[j]) + '.jpg', img)
            j = -1
        print("保存完成！")

    def segment(self):
        output = []  # 字符分割结果
        self.dataset = self.load(IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)  # 加载图像数据集和标签集
        output = self.character_segmentation(self.dataset)  # 进行字符拆分，并将字符拆分后得到的结果存储在output中
        self.image_save(output)  # 将单个字符图像连同对应标签值保存


if __name__ == '__main__':
    # output = []  # 字符分割结果
    # split = Segmentation()
    # dataset = split.load(IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)  # 加载图像数据集和标签集
    # split.image_data_show(dataset)  # 进行数据探查，显示加载的每张车牌图像
    # split.binary_show(dataset)  # 显示二值化后的每张车牌图像
    # output = split.character_segmentation(dataset)  # 进行字符拆分，并将字符拆分后得到的结果存储在output中
    # split.result_show(output)  # 显示分割得到的单个字符图像
    # split.image_save(output)  # 将单个字符图像连同对应标签值保存
    split = Segmentation()
    split.segment()
