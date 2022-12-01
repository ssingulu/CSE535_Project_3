from unicodedata import category
from flask import Flask, render_template, request
from flask import json
app = Flask(__name__)
app.config['DEBUG'] = True
import os
import shutil
import cv2
import numpy as np

# for testing
@app.route("/")
def hello():
    return "Hello Geeks!! from Google Colab"
def predict_digit(img):
    from keras.models import load_model
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    digitmodel = load_model('digitdemo1.h5')
    res = digitmodel.predict([img])[0]
    return np.argmax(res),max(res)
def classify_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    ans = None
    acc = None
    count = 0
    mx = float('-inf')
    dic = {}
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th[y - top : y + h + bottom, x - left : x + w + right]
        digit, acc = predict_digit(roi)
        if digit not in dic:
            dic[digit] = 0
        dic[digit] += 1
        if dic[digit] > mx:
            mx = dic[digit]
            ans, acc = digit, acc
        count += 1
        if count >= 42:
            break
    if ans == None:
        ans, acc = predict_digit(th)
    return ans, acc

        
@app.route("/upload_file", methods=["POST","GET"])
def upload_file():
    app.logger.info(request)
    file = request.files['image']
    file.save(file.filename)
    predicted_class, prob= classify_image(file.filename)
    predicted_class = str(predicted_class)
    isExist = os.path.exists(predicted_class)
    if not isExist:
        os.mkdir(predicted_class)
    shutil.move(file.filename, './'+predicted_class+'/'+file.filename)
    response = app.response_class(
        response=json.dumps({
            "message":"You're file is uploaded"
        }),
        status=200,
        mimetype='application/json'
    )
    return response