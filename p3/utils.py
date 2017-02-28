import numpy as np
import cv2 as cv
import csv
import random

def load_data(PATH):
    culled_list = []
    with open(PATH) as f:
        reader = csv.reader(f)
        initial_list = list(reader)

    #remove header
    initial_list.pop(0)

    #remove ones - they throw an error
    for index, item in enumerate(initial_list):
        #print(index, item[3])
        if(float(item[3])>=1):
            initial_list.pop(index)

    #cull excessive zeros - DEPRICATED - use dropout instead
    zeros =[]
    for index, item in enumerate(initial_list):
        if (float(item[3])==0.0):
            zeros.append(index)

    sample_size = float(len(zeros)) * .25
    random_zeros = random.sample(zeros, int(sample_size))

    list_of_zeros = []
    for item in random_zeros:
        list_of_zeros.append(initial_list[item])

    culled_list = [x for x in initial_list if x not in list_of_zeros]

    culled_list = initial_list;

    #reduce weak Angles DEPRICATED
    # weak_angles = []
    # for index, item in enumerate(culled_list):
    #     if (-.07 <= float(item[3]) <= .07):
    #         weak_angles.append(index)
    #
    #
    # sample_size = float(len(weak_angles)) * .35
    # random_weak = random.sample(weak_angles, int(sample_size))
    #
    # list_of_weak_angles = []
    # for item in random_weak:
    #     list_of_weak_angles.append(culled_list[item])

    #culled_list = [x for x in culled_list if x not in list_of_weak_angles]

    return culled_list

def resize_image(image):
    height = int(image.shape[0]/1)
    width = int(image.shape[1]/1)
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def flip_image(image):
    return cv.flip(image,1)

def flip_angle(angle):
    return angle*-1

def bright_image(image):
    bright_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    bright_image[:, :, 2] = bright_image[:, :, 2] * random_bright
    image1 = cv.cvtColor(bright_image, cv.COLOR_HSV2RGB)
    return image1

#DEPRICATED
def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    image_min = np.amin(image_data)
    image_max = np.amax(image_data)
    a = 0.1
    b = 0.9
    return a + ( ( (image_data - image_min)*(b - a) )/( image_max - image_min ) )

#DEPRICATED
def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    image_min = np.amin(image_data)
    image_max = np.amax(image_data)
    a = 0.1
    b = 0.9
    floats = image_data.astype(float)
    return a + ( ( (image_data - image_min)*(b - a) )/( image_max - image_min ) )

#DEPRICATED
def trans_image(image, steer, trans_range):
    # Translation
    rows, cols, channels = image.shape
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 10 * np.random.uniform() - 10 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang