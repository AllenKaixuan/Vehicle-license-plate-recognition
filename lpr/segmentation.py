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

    # 数据探查，可以显示刚才导入的图像数据集
    def image_show(self, input_image, binarize: 'bool' = False):
        if binarize:
            for img in input_image:
                reshape_img = cv.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                temp_img = self.Binarization(reshape_img)
                b_temp_img = np.copy(temp_img)
                self.Blur(b_temp_img)
                b_img = self.remove_border(b_temp_img)
                b_img = cv.resize(b_img, (300, 80))
                cv.imshow("binary_image_show", b_img)
                cv.waitKey()
            cv.destroyAllWindows()
        else:
            for img in input_image:
                cv.imshow("image_show", img)
                cv.waitKey()
            cv.destroyAllWindows()

    # 将输入的车牌图像进行灰度化和二值化操作
    def Binarization(self, input_image):
        # 灰度化该区域
        grey_image_ini = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        # 使用双边滤波方法去除噪点和二值化
        grey_image = cv.bilateralFilter(grey_image_ini, 11, 17, 17)
        ret, bin_img = cv.threshold(grey_image, 0, 255, cv.THRESH_OTSU)
        # 去除边框
        offsetX = 2
        offsetY = 1
        offset = bin_img[offsetY:-offsetY, offsetX:-offsetX]

        return offset

    # 对二值化图像中的汉字进行高斯模糊，使其在字符分割时可以被看作一个整体
    def Blur(self, input_image):
        hanzi_max_width = input_image.shape[1] // 8  # 假设汉字最大宽度为整个车牌宽度的1/8
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
        char_contours, _ = cv.findContours(
            binary_input_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 设置车牌字符的大小范围
        min_width = binary_input_image.shape[1] // 40
        min_height = binary_input_image.shape[0] * 7 // 10
        valid_char_regions = []
        # 开始对检测到的字符图形按条件进行筛选
        for i in np.arange(len(char_contours)):
            x, y, w, h = cv.boundingRect(char_contours[i])
            # 按照上面得到的字符高度和宽度进行条件筛选
            if h >= min_height and w >= min_width:
                if w < min_width*2:
                    x -= w
                    w *= 3
                #valid_char_regions.append(
                #    (x, offset_input_image[y:y + h, x:x + w]))
                #分割出来的字符常偏右上
                valid_char_regions.append(
                    (x, offset_input_image[max(int(y-h*0.2),0):y + h, x:min(int(x + w*1.1),binary_input_image.shape[1])]))
        # 将按照车牌上的x坐标从左到右排序
        sorted_regions = sorted(
            valid_char_regions, key=lambda region: region[0])

        # 将上面分割得到的字符，放到一个list中返回
        candidate_char_images = []
        for i in np.arange(len(sorted_regions)):
            candidate_char_images.append(sorted_regions[i][1])

        return candidate_char_images

    def segment(self, plates_image):
        output = []
        # 导入指定路径下的所有图像数据
        for img in plates_image:
            reshape_img = cv.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # 依次调用几个方法，对车牌图像进行字符分割
            binary_img = self.Binarization(reshape_img)
            b_temp_img = np.copy(binary_img)
            off_temp_img = np.copy(binary_img)
            # 膨胀化, 模糊化
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            b_temp_img = cv.dilate(b_temp_img, kernel)
            self.Blur(b_temp_img)
            # 去边框
            b_img = self.remove_border(b_temp_img)
            off_img = self.remove_border(off_temp_img)
            # 字符分割
            output.append(self.char_split(b_img, off_img))
        return output


if __name__ == '__main__':
    split = Segmentation()
    splited_chars = split.segment()
    split.image_save(splited_chars)
