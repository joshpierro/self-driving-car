import utils
from keras.models import load_model
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import pandas as pd
import collections
import csv

from keras.applications.resnet50 import preprocess_input, decode_predictions

model = load_model('/home/pierro/Work/udacity/SDC/p3/model.h5')
print(model.inputs)
print(model.summary())

# x = []
# image_path = img_center = 'firstimage.jpg'
# image = mpimg.imread(img_center)
#
# resized_image = utils.resize_image(image)


#i =cv.imread(image_path)
#flipped = cv.flip(i,1)
#cv.imshow( "original", i)
#cv.waitKey(0)

# x.append(image)
# x = np.array(x)

#predict = model.predict(x,batch_size=1,verbose=1)
#print(predict)
#predict2 = model.predict(x)
#print(float(predict.item(0)))

# DRIVING_DATA_PATH = 'data/driving_log.csv'
# culled_list = utils.load_data(DRIVING_DATA_PATH)

# print(len(culled_list))

# angles = []
# for item in culled_list:
#     angles.append(item[3])

#counts = collections.Counter(angles)
#
# for key,value in enumerate(counts):
#    print("angle: " + str(value), " - frequency: " + str(counts[value]))

# i =cv.imread(image_path)
# x,y = utils.trans_image(i,.1,100)
# print(y)
