import numpy as np
import cv2 as cv
import os

IMAGE_HEIGHT = 105
IMAGE_WIDTH = 400

def resize_img(img, max_size):
    """ resize图像 """
    h, w = img.shape[0:2]
    scale = max_size / max(h, w)
    img_resized = cv.resize(img, None, fx=scale, fy=scale,
                            interpolation=cv.INTER_CUBIC)

    return img_resized


def togray(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img_gray

def gb(img):
    img_gaussian = cv.GaussianBlur(img, (3, 3), 0)

    return img_gaussian


def stretching(img):
    """ 灰度拉伸 """
    maxi = float(img.max())
    mini = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 255 / (maxi - mini) * img[i, j] - (255 * mini) / (maxi - mini)
    img_stretched = img
    return img_stretched

def absdiff(img):
    """ 对开运算前后图像做差分 """
    # 进行开运算，用来去除噪声
    r = 15
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), np.uint8)
    cv.circle(kernel, (r, r), r, 1, -1)
    # 开运算
    img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # 获取差分图
    img_absdiff = cv.absdiff(img, img_opening)
    return img_absdiff


def binarization(img):
    """ 二值化处理函数 """
    maxi = float(img.max())
    mini = float(img.min())
    x = maxi - ((maxi - mini) / 2)
    # 二值化, 返回阈值ret和二值化操作后的图像img_binary
    ret, img_binary = cv.threshold(img, x, 255, cv.THRESH_BINARY)

    return img_binary

def canny(img):
    """ canny边缘检测 """
    img_canny = cv.Canny(img, img.shape[0], img.shape[1])
    return img_canny


def opening_closing(img):
    """ 开闭运算，保留车牌区域，消除其他区域，从而定位车牌 """
    # 进行闭运算
    kernel = np.ones((5, 23), np.uint8)
    img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    # 进行开运算
    img_opening1 = cv.morphologyEx(img_closing, cv.MORPH_OPEN, kernel)

    # 再次进行开运算
    kernel = np.ones((11, 6), np.uint8)
    img_opening2 = cv.morphologyEx(img_opening1, cv.MORPH_OPEN, kernel)
    return img_opening2


def find_rectangle(contour):
    """ 寻找矩形轮廓 """
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]



def locate_license(original, img):
    """ 定位车牌号 """
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_cont = original.copy()
    img_cont = cv.drawContours(img_cont, contours, -1, (255, 0, 0), 6)
    # 计算轮廓面积及高宽比
    block = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长度比
        r = find_rectangle(c)    # 里面是轮廓的左上点和右下点
        a = (r[2] - r[0]) * (r[3] - r[1])   # 面积
        s = (r[2] - r[0]) / (r[3] - r[1])   # 长度比
        block.append([r, a, s])
    # 选出面积最大的五个区域
    block = sorted(block, key=lambda b: b[1])[-5:]

    # 使用颜色识别判断找出最像车牌的区域
    maxweight, maxindex=0, -1
    for i in range(len(block)):
        # print('block', block[i])
        if 2 <= block[i][2] <=4 and 1000 <= block[i][1] <= 20000:    # 对矩形区域高宽比及面积进行限制
            b = original[block[i][0][1]: block[i][0][3], block[i][0][0]: block[i][0][2]]
            # BGR转HSV
            hsv = cv.cvtColor(b, cv.COLOR_BGR2HSV)
            lower = np.array([100, 50, 50])
            upper = np.array([140, 255, 255])
            # 根据阈值构建掩膜
            mask = cv.inRange(hsv, lower, upper)
            # 统计权值
            w1 = 0
            for m in mask:
                w1 += m / 255
                print(w1)

            w2 = 0
            for n in w1:
                w2 += n

            # 选出最大权值的区域
            if w2 > maxweight:
                maxindex = i
                maxweight = w2

    rect = block[maxindex][0]
    return rect


def preprocessing(img):
    # resize图像至300 * 400
    img_resized = resize_img(img, 400)
    # 转灰度图
    img_gray = togray(img_resized)
    # 灰度拉伸，提升图像对比度
    img_stretched = stretching(img_gray)
    # 差分开运算前后图像
    img_absdiff = absdiff(img_stretched)
    # 图像二值化
    img_binary = binarization(img_absdiff)
    # 边缘检测
    img_canny = canny(img_binary)
    # 开闭运算，保留车牌区域，消除其他区域
    img_opening2 = opening_closing(img_canny)
    # 定位车牌号所在矩形区域
    rect = locate_license(img_resized, img_opening2)
    # 框出并显示车牌
    img_copy = img_resized.copy()
    cv.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv.imshow('License', img_copy)
    return rect, img_resized

def cut_license(original, rect):
    """ 裁剪车牌 """
    license_img = original[rect[1]:rect[3], rect[0]:rect[2]]
    return license_img

if __name__ == '__main__':
    input_data = []
    label_ini = []
    label_list = []
    path = ".//image"
    plates = []
    # 导入指定路径下的所有图像数据
    for item in os.listdir(path):
        imgpath = os.path.join(path, item)
        # filename = copy.copy(item)
        if os.path.isdir(path):
            input_image_ini = cv.imread(imgpath)
            input_image = cv.resize(input_image_ini, (700, 400))
            input_data.append(input_image)

    for img in input_data:
        rect, img_resized = preprocessing(img)
        # 裁剪出车牌
        license_img = cut_license(img_resized, rect)
        plates.append(license_img)
        cv.imshow('License', license_img)
        cv.waitKey()

    cv.destroyAllWindows()

    split = Segmentation()
    split.binary_show(plates)
    splited_chars = split.segment(plates)
    split.result_show(splited_chars)

