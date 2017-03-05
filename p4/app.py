#ADVANCED LANE FINDING

#Imports
import utils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from moviepy.editor import VideoFileClip
import numpy as np

class Lanes:
    cached_result = None
    cached_lane = None
    last_good_right_curve = None
    last_good_left_curve = None

def pipeline(video):
    print('processing')
    masked = utils.mask_image(video)
    undistorted_dash = cv.undistort(masked, cam_mtx, cam_dist, None, cam_mtx)
    m = cv.getPerspectiveTransform(utils.source(), utils.destination())
    m_inverse = cv.getPerspectiveTransform(utils.destination(),utils.source())
    image_size = (undistorted_dash.shape[1],undistorted_dash.shape[0])
    warped = cv.warpPerspective(undistorted_dash, m, image_size, flags=cv.INTER_LINEAR)
    threshold = utils.get_threshold(warped)

    histogram = np.sum(threshold[threshold.shape[0]/2:,:], axis=0)
    out_img = np.dstack((threshold, threshold, threshold))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
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

# this try catch culls out bad frames
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Fit a second order polynomial to each
        Lanes.last_good_right = right_fit
    except:
        return Lanes.cached_result

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

    ####
    # calculate curves and centerline
    left_curverad, right_curverad =  utils.calculate_curves(leftx, lefty, rightx, righty)
    center_calc = utils.get_center_calc(video,left_fitx,right_fitx)


    ###
    warp_zero = np.zeros_like(threshold).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # this is a hamfisted solution to replace curves with extreme values with the previous frame's curve
    if (right_curverad > 5000 or left_curverad > 5000) and Lanes.cached_lane is not None:
        newwarp = Lanes.cached_lane
    else:
        newwarp = cv.warpPerspective(color_warp, m_inverse, (video.shape[1], video.shape[0]))
        Lanes.cached_lane = newwarp
        Lanes.last_good_right_curve = right_curverad
        Lanes.last_good_left_curve = left_curverad

    result = cv.addWeighted(video, 1, newwarp, 0.3, 0)
    Lanes.cached_result = result

    text = 'curvature radius: {0} m. '.format((int(left_curverad) + int(right_curverad)) / 2 )
    text2 = 'distance from center: {0} m. '.format(( np.math.ceil(abs(center_calc) * 100) / 100))
    cv.putText(result,text,(25,75),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2,cv.LINE_AA)
    cv.putText(result, text2, (25, 120), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv.LINE_AA)

    return result

#calibrate Camera
cam_mtx, cam_dist = utils.calibrate_camera()

#process video
# video_path = 'project_video.mp4'
# video_output = 'project_result.mp4'
# output = VideoFileClip(video_path)
# input = output.fl_image(pipeline)
# input.write_videofile(video_output, audio=False)


##rubric point 1
# image = cv.imread('camera_cal/calibration1.jpg')
# undist = cv.undistort(image, cam_mtx, cam_dist, None, cam_mtx)
#
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


# #rubric point 2
# image = cv.imread('test_images/test1.jpg')
# masked = utils.mask_image(image)
# undistorted_dash = cv.undistort(masked, cam_mtx, cam_dist, None, cam_mtx)
#
# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(image,cmap='gray')
# plt.xlabel('Original Image')
# plt.xticks([], [])
# plt.yticks([], [])
#
# plt.subplot(1, 2, 2)
# plt.imshow(undistorted_dash)
# plt.xlabel('Undistorted and Masked Image')
# plt.xticks([], [])
# plt.yticks([], [])
# plt.show(block=True)
#

# #point 3
# image = cv.imread('test_images/test1.jpg')
# masked = utils.mask_image(image)
# undistorted_dash = cv.undistort(masked, cam_mtx, cam_dist, None, cam_mtx)
# m = cv.getPerspectiveTransform(utils.source(), utils.destination())
# m_inverse = cv.getPerspectiveTransform(utils.destination(), utils.source())
# image_size = (undistorted_dash.shape[1], undistorted_dash.shape[0])
# warped = cv.warpPerspective(undistorted_dash, m, image_size, flags=cv.INTER_LINEAR)
#
# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(undistorted_dash)
# plt.xlabel('Undistorted Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(warped)
# plt.xlabel('warped Image')
# plt.show(block=True)

# #point 4
# image = cv.imread('test_images/test2.jpg')
# masked = utils.mask_image(image)
# undistorted_dash = cv.undistort(masked, cam_mtx, cam_dist, None, cam_mtx)
# m = cv.getPerspectiveTransform(utils.source(), utils.destination())
# m_inverse = cv.getPerspectiveTransform(utils.destination(), utils.source())
# image_size = (undistorted_dash.shape[1], undistorted_dash.shape[0])
# warped = cv.warpPerspective(undistorted_dash, m, image_size, flags=cv.INTER_LINEAR)
# threshold = utils.get_threshold(warped)
#
# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(warped)
# plt.xlabel('Undistorted Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(threshold,cmap='gray')
# plt.xlabel('Threshold Image')
# plt.show(block=True)

#point 5
# image = cv.imread('test_images/test2.jpg')
# masked = utils.mask_image(image)
# undistorted_dash = cv.undistort(masked, cam_mtx, cam_dist, None, cam_mtx)
# m = cv.getPerspectiveTransform(utils.source(), utils.destination())
# m_inverse = cv.getPerspectiveTransform(utils.destination(), utils.source())
# image_size = (undistorted_dash.shape[1], undistorted_dash.shape[0])
# warped = cv.warpPerspective(undistorted_dash, m, image_size, flags=cv.INTER_LINEAR)
# threshold = utils.get_threshold(warped)
# histogram = np.sum(threshold[threshold.shape[0] / 2:, :], axis=0)
# out_img = np.dstack((threshold, threshold, threshold)) * 255
#
# plt.plot(histogram)
# plt.show(block=True)

# image = cv.imread('test_images/test2.jpg')
# pipeline(image)
#out_img = pipeline(image)


# plt.subplot(1, 2, 2)
# plt.imshow(color_warp)
# plt.xlabel('warp with drawing')
# plt.show(block=True)


# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show(block=True)


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


# plt.subplot(1, 2, 2)
# plt.imshow(threshold, cmap='gray')
# plt.xlabel('Threshold Image')
# plt.show(block=True)


# #Read Video
# white_output = 'output_videos/project_output.mp4'
# project_video = VideoFileClip('project_video.mp4')
# #white_clip = clip1.fl_image(process_image)
# #white_clip.write_videofile(white_output, audio=False)

