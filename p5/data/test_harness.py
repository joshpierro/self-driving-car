import glob

import utils
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



#CONSTANTS
VEHICLE_IMAGES = 'data/vehicles/**/*.png'
NON_VEHICLE_IMAGES = 'data/non-vehicles/**/*.png'

#load data
# vehicle_data = utils.load_data(VEHICLE_IMAGES)
# non_vehicle_data = utils.load_data(NON_VEHICLE_IMAGES)
# v = plt.imread(vehicle_data[999])
# n = plt.imread(non_vehicle_data[999])
#
# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(v)
# plt.xlabel('Vehicle')
#
# plt.subplot(1, 2, 2)
# plt.imshow(n)
# plt.xlabel('non-Vehicle')
# plt.show(block=True)


# # vehicle_data = vehicle_data[0:100]
# # non_vehicle_data = non_vehicle_data[0:100]
#
# print(len(vehicle_data))
# print('loaded non vehicle images')
#
#
# image = mpimg.imread('/home/pierro/Work/udacity/self-driving-car/p5/test_images/test1.jpg',0)
#
#
# image = mpimg.imread('/home/pierro/Work/udacity/self-driving-car/p5/test_images/test1.jpg',0)
# windows = utils.slide_window(image, x_start_stop=[None,None], y_start_stop=[360,650],
#                     xy_window=(128, 128), xy_overlap=(0.5, 0.5))
# print(image.shape)
# draw_image = np.copy(image)
#
#
# window_img = utils.draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
#
#
# plt.imshow(window_img)
# plt.xlabel('windows')
# plt.show(block=True)