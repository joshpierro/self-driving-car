#ADVANCED LANE FINDING

#Imports
import utils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

import numpy as np

#calibrate Camera
cam_mtx, cam_dist = utils.calibrate_camera()

image = mpimage.imread('test_images/test6.jpg')
undistorted_dash = cv.undistort(image, cam_mtx, cam_dist, None, cam_mtx)


src = utils.source()
dest = utils.destination()
matrix = cv.getPerspectiveTransform(src, dest)
image_size = (undistorted_dash.shape[1],undistorted_dash.shape[0])
print(image_size)

warped = cv.warpPerspective(image, matrix, image_size, flags=cv.INTER_LINEAR)

threshold = utils.get_threshold(warped)
histogram = np.sum(threshold[threshold.shape[0]/2:,:], axis=0)
out_img = np.dstack((threshold, threshold, threshold))*255

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
window_height = np.int(threshold.shape[0]/nwindows)
nonzero = threshold.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
leftx_current = leftx_base
rightx_current = rightx_base
margin = 100
minpix = 50
left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = threshold.shape[0] - (window+1)*window_height
    win_y_high = threshold.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
    cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, threshold.shape[0]-1, threshold.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show(block=True)


# histogram = np.sum(threshold[threshold.shape[0]/2:,:], axis=0)
# plt.plot(histogram)
# plt.show(block=True)

# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(undistorted_dash)
# plt.xlabel('Undistorted Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(threshold, cmap='gray')
# plt.xlabel('threshold warped Image')
# plt.show(block=True)


# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(undistorted_dash)
# plt.xlabel('Undistorted Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(warped)
# plt.xlabel('warped Image')
# plt.show(block=True)

# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.xlabel('Original Image')
# plt.xticks([], [])
# plt.yticks([], [])
#
# plt.subplot(1, 2, 2)
# plt.imshow(undist)
# plt.xlabel('Undistorted Image')
# plt.xticks([], [])
# plt.yticks([], [])
# plt.show(block=True)

# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(undistorted_dash)
# plt.xlabel('Undistorted Image')
#
#
# plt.subplot(1, 2, 2)
# plt.imshow(aerial)
# plt.xlabel('Aerial Image')
# plt.show(block=True)


# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(undist)
# plt.xlabel('Undistorted Image')

#
# plt.subplot(1, 2, 2)
# plt.imshow(threshold, cmap='gray')
# plt.xlabel('Threshold Image')
# plt.show(block=True)


# #Read Video
# white_output = 'output_videos/project_output.mp4'
# project_video = VideoFileClip('project_video.mp4')
# #white_clip = clip1.fl_image(process_image)
# #white_clip.write_videofile(white_output, audio=False)

