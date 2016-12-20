import cv2
import numpy as np
import Constants

class MaskCoordinates():
    nw_x=0
    nw_y=0
    nw_x2 = 0
    nw_y2 = 0
    ne_x=0
    ne_y=0
    ne_x2 = 0
    ne_y2 = 0
    sw_x=0
    sw_y=0
    sw_x2 = 0
    sw_y2 = 0
    se_x=0
    se_y=0
    se_x2 = 0
    se_y2 = 0


def set_mask_bounds(shape):
    """returns an object that hold the bounding coordinates of the image to be processed
    TODO - make constants for these values
    """
    mask_coordinates = MaskCoordinates()
    mask_coordinates.sw_x = shape[1]*.01
    mask_coordinates.sw_y = int(shape[0])
    mask_coordinates.sw_x2 = shape[1] * .30
    mask_coordinates.sw_y2 = int(shape[0])
    mask_coordinates.nw_x = int(shape[1]*.35)
    mask_coordinates.nw_y = int(shape[0]*.65)
    mask_coordinates.nw_x2 = int(shape[1] * .50)
    mask_coordinates.nw_y2 = int(shape[0] * .65)
    mask_coordinates.ne_x = int(shape[1]*.65)
    mask_coordinates.ne_y = int(shape[0]*.65)
    mask_coordinates.ne_x2 = int(shape[1]*.55)
    mask_coordinates.ne_y2 =  int(shape[0]*.65)
    mask_coordinates.se_x = int(shape[1] + (shape[1]*.1) )
    mask_coordinates.se_y = int(shape[0] )
    mask_coordinates.se_x2 = int(shape[1] - (shape[1]*.5) )
    mask_coordinates.se_y2 = int(shape[0])
    return mask_coordinates

def get_mask_bounds(mask_bounds):
    return np.array([[(mask_bounds.sw_x, mask_bounds.sw_y),
                          (mask_bounds.nw_x,mask_bounds.nw_y),
                          (mask_bounds.nw_x2  , mask_bounds.nw_y2),
                          (mask_bounds.sw_x2, mask_bounds.sw_y2),
                          (mask_bounds.se_x2, mask_bounds.se_y2),
                          (mask_bounds.ne_x2,mask_bounds.ne_y2),
                          (mask_bounds.ne_x,mask_bounds.ne_y),
                          (mask_bounds.se_x,mask_bounds.se_y)]], dtype=np.int32)

def region_of_interest(image, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def grayscale(image):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def increase_contrast(image):
    """this function increases the contrast of the BW image
    TODO learn more about what is happening here
    """
    equ = cv2.equalizeHist(image)
    hist, bins = np.histogram(equ.flatten(), Constants.bin_256, [Constants.hist_range_min, Constants.hist_range_max])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, Constants.masked_array_min)
    cdf_m = (cdf_m - cdf_m.min()) * Constants.masked_array_max / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, Constants.masked_array_min).astype('uint8')
    histogram = cdf[image]
    return cv2.equalizeHist(histogram)

class HoughParamters():
    """this is an object used to hold the calculated houh parameters"""
    rho = 0
    min_line_length = 0
    max_line_gap = 0
    threshold = 0
    theta = 0

def set_hough_paramters(image):
    """this function calculates the hough parameters with its image size"""
    hough_params = HoughParamters()
    hough_params.rho = (image.shape[0] * image.shape[1]) * Constants.rho_multiplier
    hough_params.min_line_length = (image.shape[0] * image.shape[1]) *  Constants.min_line_length_multiplier
    hough_params.max_line_gap = (image.shape[0] * image.shape[1]) * Constants.max_line_gap_multiplier
    hough_params.threshold = int((image.shape[0] * image.shape[1]) * Constants.threshold_multiplier)
    hough_params.theta = Constants.theta
    return hough_params


def canny(image, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(image, low_threshold, high_threshold)


def gaussian_blur(image, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)



class Line_object():
    line=[]
    slope=1
    slope_is_positive = False
    draw_line = False

def set_up_line(line):
    """this function sets up a line object for processing"""
    line_object = Line_object()
    line_object.line = line
    for x1, y1, x2, y2 in line_object.line:
        line_object.slope =  (y2 - y1) / (x2 - x1)
    if(line_object.slope>0):
        line_object.slope_is_positive = True
    return line_object



def  vett_lines(line_object):
    """this function flags lines that are appropriate for lane markings
    first it takes a a quick swag by culling gentle sloped lines,
    then it only flags lines that are within a certain standard deviation
    """
    if line_object.slope_is_positive == False and (-.8 <= line_object.slope <= -.55):
        Constants.average_slope_counter = Constants.average_slope_counter+1
        if(line_object.slope!=0):
            Constants.average_left.append(line_object.slope)
            left_slope_standard_deviation = np.std(Constants.average_left);
            if(left_slope_standard_deviation + (line_object.slope * left_slope_standard_deviation) <=.03):
                Constants.good_left_line = line_object.line
                line_object.draw_line = True
    if line_object.slope_is_positive == True and (.55 <= line_object.slope <= .8):
        if (line_object.slope != 0):
            Constants.average_right.append(line_object.slope)
            right_slope_standard_deviation = np.std(Constants.average_right);
            if ( right_slope_standard_deviation - (line_object.slope * right_slope_standard_deviation) <= .03):
                Constants.good_right_line = line_object.line
                line_object.draw_line = True
        line_object.draw_line = True
    return line_object


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).

    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        line_object = set_up_line(line)
        vetted_line = vett_lines(line_object)
        #
        if (vetted_line.draw_line == True):
            for x1, y1, x2, y2 in vetted_line.line:
                # TODO move extension to a function
                if vetted_line.slope_is_positive == False:
                    # Extend back
                    x1 = int((image.shape[0] - y2) / vetted_line.slope) + x2
                    y1 = image.shape[0]
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                    # extend front
                    x1_2 = x2
                    y1_2 = y2
                    y2_2 = int(image.shape[0] * .65)
                    x2_2 = int((y2_2 - y1_2) / (vetted_line.slope)) + x1_2
                    cv2.line(image, (x1_2, y1_2), (x2_2, y2_2), color, thickness)


                if vetted_line.slope_is_positive == True:
                    # extend Back
                    y2 = image.shape[0]
                    x2 = int((image.shape[0] - y1) / vetted_line.slope) + x1
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                    # extend front
                    x1_2 = x2
                    y1_2 = y2
                    y2_2 = int(image.shape[0] * .65)
                    x2_2 = int((y2_2 - y1_2) / (vetted_line.slope)) + x1_2
                    cv2.line(image, (x1_2, y1_2), (x2_2, y2_2), color, thickness)




def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if lines != None:
        draw_lines(line_img, lines,[255, 0, 0],15)

    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(image, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, image, β, λ)