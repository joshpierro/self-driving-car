import glob

import cv2 as cv
import numpy as np

#CONSTANTS
NX = 9 #inside corners in x
NY = 6 #inside corners in y
CALIBRATION_IMAGES = glob.glob('camera_cal/calibration*.jpg')

THRESH_MIN = 20
THRESH_MAX = 100

#calibrate camera
def calibrate_camera ():

    object_points = []
    image_points = []

    for calibration_image in CALIBRATION_IMAGES:
        image = cv.imread(calibration_image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (NX, NY), None)

        if ret:
            objp = np.zeros((NX * NY, 3), np.float32)
            objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)
            object_points.append(objp)
            image_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return mtx,dist


#get sobel
def get_sobel(image):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= THRESH_MIN) & (scaled_sobel <= THRESH_MAX)] = 1
    return sxbinary



#get threshold
def get_threshold(image):
    hls = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2HLS)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    s = hls[:, :, 2]

    x, gray_threshold = cv.threshold(gray.astype('uint8'), 150, 255, cv.THRESH_BINARY)
    x, s_threshold = cv.threshold(s.astype('uint8'), 150, 255, cv.THRESH_BINARY)

    combined_binary = np.clip(cv.bitwise_and(gray_threshold, s_threshold), 0, 1).astype('uint8')

    return combined_binary

#
def source():
    src = np.float32([
        [475,530],
        [830,530],
        [130,720],
        [1120,720]
    ])
    return src

def destination():
    src = np.float32([
        [365,540],
        [990,540],
        [320,720],
        [960,720]
    ])
    return src





# combined_binarydef get_warped_perspective(img, reverse_persp=False):
#
#     src = get_src(w, h)
#     dest = get_dest(w, h)
#     if reverse_persp:
#         matrix = cv.getPerspectiveTransform(dest, src)
#     else:
#         matrix = cv.getPerspectiveTransform(src, dest)
#     flipped = img.shape[0:2][::-1]
#     return cv.warpPerspective(img, matrix, flipped)
#
#
# def src(w, h):
#     return
#
#
#
# def dest(w, h):


