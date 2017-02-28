import utils
import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

import matplotlib.image as mpimg

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        #steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image.save('/home/pierro/Work/udacity/SDC/p3/image.jpg')

        #this is a hack - TODO: clean up
        image_path  = '/home/pierro/Work/udacity/SDC/p3/image.jpg'
        image = mpimg.imread(image_path)

        #resize
        image = utils.resize_image(image)

        x = []
        x.append(image)
        x = np.array(x)

        #predict
        predict = model.predict(x, batch_size=1, verbose=1)

        steering_angle = float(predict.item(0))
        throttle = 0.2
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        help='model.h5'
    )

    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )


    args = parser.parse_args()
    model = load_model('/home/pierro/Work/udacity/SDC/p3/model.h5')
    #model = load_model('/home/pierro/Work/udacity/SDC/p3/checkpoints/model07.h5') #6

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)