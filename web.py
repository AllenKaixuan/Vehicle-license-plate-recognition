import datetime
import flask
import json
import cv2
import os
from flask import Flask, request, make_response
import recognize

sever = Flask(__name__)


@sever.route("/upload", methods=["post"])
def upload():
    f = flask.request.files.get('file', None)
    if f:  # 如果文件不为空
        cur_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = str(f.filename)
        name = filename[:filename.rfind('.')]
        suffix = filename[filename.rfind('.'):]

        new_file_name = name + cur_time + suffix
        print('new_file_name', new_file_name)

        file_path = os.path.join('cars', new_file_name)
        f.save(file_path)

        info = scan_image(new_file_name)
        res = {"msg": "文件上传成功", 'data': info}
    else:   # 如果文件为空
        res = {"msg": "没有上传文件"}
    return json.dumps(res, ensure_ascii=False)  # 防止出现乱码


def scan_image(file_path):
    img = cv2.imread(file_path)
    # 识别结果
    cars = LPR(img)
    print(cars)

    arr = []
    for car in cars:
        label = car[0]
        confidence = car[1]
        point = car[2]
        cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]),
                      (0, 0, 255), 2)
        arr.append({'label': label, 'confidence': confidence, 'point': point})
    cv2.imwrite(file_path, img)
    url = '/' + file_path.replace('\\', '/')
    carInfo = {'filename': file_path, 'url': url, 'cars': arr}
    return carInfo


@sever.route('/cars/<string:filename>', methods=['GET'])
def show_photo(filename):
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join('cars', filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


sever.run(port=8888)
