import Utils as utils
import Constants as constants
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import cv2

import os

print('Hello Lane Lines')

def process_image(image):
    print("processing")
    mask_bounds = utils.set_mask_bounds(image.shape)
    mask_vertices = utils.get_mask_bounds(mask_bounds)
    masked_image = utils.region_of_interest(image, mask_vertices)
    gray_image = utils.grayscale(masked_image)
    contrast_image = utils.increase_contrast(gray_image)
    range = cv2.inRange(contrast_image, constants.in_range_min, constants.in_range_max)
    edges = utils.canny(range, constants.low_threshold, constants.high_threshold)
    hough_parameters = utils.set_hough_paramters(image)
    lines = utils.hough_lines(edges, hough_parameters.rho, hough_parameters.theta, hough_parameters.threshold,
                              hough_parameters.min_line_length, hough_parameters.max_line_gap)
    return utils.weighted_img(lines, image)


videos = os.listdir("videos/source")
for video in videos:
    print('processing' + video)
    video_path = "videos/source/" + video
    video_output = "videos/output/" + video + "_output.mp4"
    output = VideoFileClip(video_path)
    input = output.fl_image(process_image)
    input.write_videofile(video_output, audio=False)

print("Videos ready at: /videos/output")

"""test images
image = mpimg.imread('images/solidWhiteRight.jpg')
g = process_image(image)
plt.imshow(g, cmap='gray')
plt.show(block=True)

image = mpimg.imread('images/challenge.jpg')
g = process_image(image)
plt.imshow(g, cmap='gray')
plt.show(block=True)


image = mpimg.imread('images/challenge2.jpg')
g = process_image(image)
plt.imshow(g, cmap='gray')
plt.show(block=True)

image = mpimg.imread('images/solidYellowCurvejpg')
g = process_image(image)
plt.imshow(g, cmap='gray')
plt.show(block=True)
"""




"""
Videos

white_output = 'videos/output/white_output.mp4'
clip1 = VideoFileClip("videos/source/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

white_output = 'videos/output/yellow_output.mp4'
clip1 = VideoFileClip("videos/source/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

white_output = 'videos/output/challenge_output.mp4'
clip1 = VideoFileClip("videos/source/challenge.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

white_output = 'videos/output/brooklyn.mp4'
clip1 = VideoFileClip("videos/source/brooklyn.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

"""



