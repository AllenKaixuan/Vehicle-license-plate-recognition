import datetime
from unittest import result
import flask
import json
import cv2
import os
from flask import Flask, request, make_response, render_template, url_for
import lpr.recognize as recognize

sever = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))


@sever.route("/upload", methods=["post"])
def upload():
    f = flask.request.files.get('upload_image', None)
    if f:  # 如果文件不为空
        cur_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = str(f.filename)
        name = filename[:filename.rfind('.')]
        suffix = filename[filename.rfind('.'):]

        new_file_name = name + cur_time + suffix
        print('new_file_name', new_file_name)

        file_path = os.path.join('static/images', new_file_name)
        f.save(file_path)

        info = scan_image(file_path)
        res = {"msg": "文件上传成功", 'data': info}
        return render_template('index.html',
                               img_url=url_for('static',filename='images/'+new_file_name),
                               plate=info,
                               )
    else:   # 如果文件为空
        res = {"msg": "没有上传文件"}
    return json.dumps(res, ensure_ascii=False)  # 防止出现乱码


def scan_image(file_path):
    img = cv2.imread(file_path)
    # 识别结果
    result = recognize.recognize(img)
    return result

    #arr = []
    # for car in result:
    #    label = car[0]
    #    confidence = car[1]
    #    arr.append({'label': label, 'confidence': confidence})
    #cv2.imwrite(file_path, img)
    #url = '/' + file_path.replace('\\', '/')
    #carInfo = {'filename': file_path, 'url': url, 'cars': arr}
    # return carInfo


# @sever.route('/cars/<string:filename>', methods=['GET'])
# def show_photo(filename):
#    if request.method == 'GET':
#        if filename is None:
#            pass
#        else:
#            image_data = open(os.path.join('cars', filename), "rb").read()
#            response = make_response(image_data)
#            response.headers['Content-Type'] = 'image/png'
#            return response
#    else:
#        pass


sever.run(port=8888)
