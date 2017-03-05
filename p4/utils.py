import glob

import cv2 as cv
import numpy as np

#CONSTANTS
NX = 9 #inside corners in x
NY = 6 #inside corners in y
CALIBRATION_IMAGES = glob.glob('camera_cal/calibration*.jpg')

THRESH_MIN = 20
THRESH_MAX = 100

BIN_256 = 256
HIST_RANGE_MIN = 0
HIST_RANGE_MAX = 256
MASKED_ARRAY_MIN = 0
MASKED_ARRAY_MAX = 255

YM_PER_PX = 30 / 720
XM_PER_PX = 3.7 / 700

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

#mask
def mask_image(image):
    sw_x = image.shape[1] * .01
    sw_y = image.shape[0]
    nw_x = int(image.shape[1] * .35)
    nw_y = int(image.shape[0] * .65)
    ne_x = int(image.shape[1] * .65)
    ne_y = int(image.shape[0] * .65)
    se_x = int(image.shape[1] + (image.shape[1] * .1))
    se_y = int(image.shape[0])


    vertices = np.array([[(sw_x, sw_y),
                          (nw_x,nw_y),
                          (ne_x,ne_y),
                          (se_x,se_y)]], dtype=np.int32)



    mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

def increase_contrast(image):
    """this function increases the contrast of the BW image
    """
    equ = cv.equalizeHist(image)
    hist, bins = np.histogram(equ.flatten(), BIN_256, [HIST_RANGE_MIN, HIST_RANGE_MAX])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, MASKED_ARRAY_MIN)
    cdf_m = (cdf_m - cdf_m.min()) * MASKED_ARRAY_MAX / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, MASKED_ARRAY_MIN).astype('uint8')
    histogram = cdf[image]
    return cv.equalizeHist(histogram)


#get threshold
def get_threshold(image):
    hls = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2HLS)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = increase_contrast(gray)
    s = hls[:, :, 2]
    x, gray_threshold = cv.threshold(gray.astype('uint8'), 75, 255, cv.THRESH_BINARY)
    x, s_threshold = cv.threshold(s.astype('uint8'), 75, 255, cv.THRESH_BINARY)
    combined_binary = np.clip(cv.bitwise_and(gray_threshold, s_threshold), 0, 1).astype('uint8')
    return combined_binary


# thresh_min = 20
# thresh_max = 100
# gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
# abs_sobelx = np.absolute(sobelx)
# scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
# sxbinary = np.zeros_like(scaled_sobel)
# sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
# return sxbinary


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


#calculate curves
def calculate_curves(leftx,lefty,rightx,righty):
    left_fit_cr = np.polyfit(lefty * YM_PER_PX, leftx * XM_PER_PX, 2)
    right_fit_cr = np.polyfit(righty * YM_PER_PX, rightx * XM_PER_PX, 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(lefty) + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(lefty) + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    return left_curverad,right_curverad

def get_center_calc(video,left_fitx,right_fitx):
    vehicle_position = video.shape[1] / 2
    center_of_lane = (left_fitx[-1] + right_fitx[-1]) // 2
    center_calc = (vehicle_position - center_of_lane) * XM_PER_PX
    return center_calc

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


