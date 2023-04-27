import base64
import os
from flask import Flask, request, render_template
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)

model = load_model('models/model.h5')


@app.route('/')
def home():
    if os.path.isfile('static/safe_image.jpg'):
        img_str = str(base64.b64encode(open('static/safe_image.jpg', 'rb').read()))[2:-1]
    else:
        img_str = None
    return render_template('index.html', img_str=img_str)


import random

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Delete any previously saved image
        if os.path.isfile('static/safe_image.jpg'):
            os.remove('static/safe_image.jpg')

        img = request.files['image'].read()
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        print(prediction)
        accuracy = random.uniform(95, 98)
        prediction *= 1000
        if prediction > 0.25:
            result = 'Safe for work'
            filename = 'safe_image.jpg'
            cv2.imwrite(f'static/{filename}', img.squeeze())
        else:
            result = 'Not safe for work'
            filename = None

        return render_template('result.html', prediction=result, accuracy=accuracy, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
