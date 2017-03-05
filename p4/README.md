**Advanced Lane Finding Project**

This repo contains my submission for the <strong>Advanced Lane Finding</strong> project, and this readme describes my approach and its components. The subsequent sections outline how these components fulfill the requirements defined in the <a href='https://review.udacity.com/#!/rubrics/571/view'>project rubric</a>

My app.py file contains all of the code that defines and execute my pipeline. Methods in the utils.py script were used to delegate much of the logic used to process the images.

point1.png	add calibration image for write up	6 hours ago
point2.png	add image for undistorted image	6 hours ago
point3.png	add images for birdseye view	5 hours ago
point4.png	add threshold image	4 hours ago
point5.png	add histogram and sliding window images	4 hours ago
point5_1.png	add histogram and sliding window images	4 hours ago
point6.png	add screenshot and video for final output	an hour ago
save_output_here.txt

[//]: # (Image References)
[point1]: ./output_images/point1.png "Camera Calibration/Undistorted Imagwe"
[point2]: ./output_images/point2.png "Undistorted Image"
[point3]: ./output_images/point3.png "Perspective Transform"
[point4]: ./output_images/point4.png "Threshold/Binary image"
[point5]: ./output_images/point5.png "Histogram"
[point5_1]: ./output_images/point5_1.png "Sliding Window"
[point6]: ./output_images/point6.png "Final Result"
[bad_result]: ./output_images/bad_result.png "Bad results"


###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The purpose of this exercise is to correct any distortion that is introduced by the car's camera, which occurs when objects in the real world (3d) are projected on to a 2d plane. Several variated images of a chessboard pattern are used to define and correct the camera's distortion in the calibration process.   

In the calibrate_camera function, the arrays 'object_points' and 'image_points' are defined to hold the coordinates of  'chessboard' corners in real world space (3d) and 2d space. Next, all of the calibration images are looped through, where:
* the image is converted to gray scale
* each grayscale image is passed to the OpenCV method 'findChessboardCorners', along with the number of corners (NX,NY) 
* The corners of each grayscale image are added to the image_points array

Once the calibration images have been processed 'object_points' and 'image_points' are passed to the OpenCV method calibrateCamera, which returns the camera matrix, distortion coefficients, rotation and translation vectors. These values are subsequently used in a function (cv.undistort) that undistorts the car's camera. 

The following image illustrates an chessboard, before and after calibration and processing it with the OpenCV undistort function. App.py lines 143-144 contain the code that generated this image. 
<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point1.png'>

The function for calibrating my camera (calibrate_camera) can be found in utils.py on lines 24-41. It is called by app.py on line 132 after my pipeline is defined.  

<pre>
#calibrate Camera<br>
cam_mtx, cam_dist = utils.calibrate_camera()
</pre>

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Using the output of the previous step, I was able to generate an undistorted image using the OpenCV 'undistort' method. 
This can be seen on line 20 of app.py.  

<pre>
    undistorted_dash = cv.undistort(masked, cam_mtx, cam_dist, None, cam_mtx)
</pre>

It should be noted that I also applied a mask to the original image to remove un-needed pixels and aid in future processing. An example of a distortion corrected image can be found below. The correction is most visible along the bottom edge. The code that generated these images can be found in lines 162-178 in app.py. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point2.png'>


####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

When I first attempted the perspective transform, I took an image from google maps and a test image (test1.jpg) and tried to define my source and destination control points to be used in the OpenCV perspective transform function. This approach was inspired by the stop sign exercise in the lectures. This, however, gave wonky results and I was unable to successfully get an acceptable birds eye image.  

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/aerial.png'>

As a result, I ended up using hard coded values for my source and destination that were derived through lots of trial and error. The values for my source and destination can be found in lines 124-140 in utils.py and my implementation can be found on line 21-24 in app.py. It should be noted that I also derived an inverse perspective transform for when the birds eye view is converted back to the dash cam view.  

<pre>
    m = cv.getPerspectiveTransform(utils.source(), utils.destination())
    m_inverse = cv.getPerspectiveTransform(utils.destination(),utils.source())
    image_size = (undistorted_dash.shape[1],undistorted_dash.shape[0])
    warped = cv.warpPerspective(undistorted_dash, m, image_size, flags=cv.INTER_LINEAR)
</pre>

An example of my perspective transform can be found below. The code that generated these images can be found on lines 181-198 in app.py. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point3.png'>


####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of thresholds in the S-channel of HLS color space and grayscale to derive a binary image from my warped image. The OpenCV 'threshold' method with cv.THRESH_BINARY as the threshold type was ultimately used. I found that this gave a slightly better result than the OpenCV sobel method. My implementation can be found on line 25 in app.py, which calls the 'get_threshold' function in utils.py (lines 101-109). 

<pre>threshold = utils.get_threshold(warped)</pre>

The following image illustrates the threshold binary derived from a warped image. The code to generate this image can be found on lines 201-218. 
<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point4.png'>

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the histogram and sliding window method, as described in the lectures, to identify my lane lines and fit their position with a polynomial. 

The first step was to take a histogram along the lower 1/2 of the image (line 27 in app.py)

<pre>  histogram = np.sum(threshold[threshold.shape[0]/2:,:], axis=0)</pre>
<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point5.png'>

Next, the peaks of the right and left halves of the histogram are identified, which will be used as the starting point for the lanes (lines 30-32 in app.py). 
<pre>
    midpoint = np.int(histogram.shape[0]/2)<br>
    leftx_base = np.argmax(histogram[:midpoint])<br>
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
</pre>

After the baselines for the lane lines are identified, a sliding window technique is applied to discover the rest of the lane pixels. This technique divides the image into horizontal sections (9 was chosen in my case), where baselines are derived and lane lines can be found and followed up to the top of the frame. Lines 37-70 in app.py is where the sliding window parameters are set up and it loops through the windows. 

Once the left and right lane line pixels have been extracted, a second order polynomial is applied to each. The image below illustrates the sliding window and polynomial overlays on a binary lane line image. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point5_1.png'>


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature of the lane lines were calculated with the algorithm provided in the class material. It plots a second order polynomial to pixel positions for each lane line, then calculates the radius. My implementation can be found on line 101 of app.py, with the details of the implementation found on lines 144-150 (calculate_curves) of utils.py.

<pre>left_curverad, right_curverad =  utils.calculate_curves(leftx, lefty, rightx, righty)</pre>

The distance of the car in relation to the center of the lane was derived by assuming the center of the image was the center of the car, then calculating how far value that was from each lane boundary. Line 102 in app.py calls the function get_center_calc lines 152-156  The average of those two values were displayed. 

<pre>center_calc = utils.get_center_calc(video,left_fitx,right_fitx)</pre>

Both operations used the following assumptions to convert pixels to meters:

<pre>
YM_PER_PX = 30 / 720<br>
XM_PER_PX = 3.7 / 700
</pre>


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Lines 105-127 in app.py handle rendering the lane lines, performing a (reverse) perspective transform, adding annotation to the video frame and returning the result. A screen shot of the result can be found below:

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/point6.png'>


###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are OK but no catastrophic failures that would cause the car to drive off the road!).

My final video can be found in this repo's <a href='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_videos/project_video.mp4'>output_videos</a> directory, or on <a href='https://youtu.be/sAbNnXe8I-M'>youtube</a>. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/sAbNnXe8I-M" frameborder="0" allowfullscreen></iframe>

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I was pleasantly surprised to find that if one followed the guidelines in the lectures, one could get solution that was 80% there fairly easily. As always, my approach to these sorts of problems is to get an end to end solution and iterate until the result was acceptable. That said, a fair amount of trial and error was necessary for each step and a few roadblocks were encountered along the way. 

The first impediment I found was that the sliding window implementation produced some zero length arrays that blew up the function that fit the lane lines with a polynomial. To overcome this I simply cached the last good frame result and fell back on that if zero length arrays were found. 

The only other real problem I had was that in certain sections of road there were very large values in the lane indices that resulted in angles/lane lines that had unpredictable results. To combat this I manually identified a threshold for acceptable values in the indices and if a frame did not meet that limit I fell back to the last acceptable result. While this is admittedly a ham fisted solution, it provided decent results on the project_video. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p4/output_images/bad_result.png'>

This approach, however, failed miserably on the subsequent challenge videos with sharp turns. The steep curves regularly exceeded the thresholds that I set. More exploration is needed to identify and address the root cause of the outliers on normal straight road conditions to come up with a better solution.

Currently, however, my pipeline has proven to perform poorly with videos/images that have sharp curves. Other anomalies are also likely to make it fail as well, such as different lighting conditions, adverse weather conditions, unusual road conditions/surfaces and more  

Ideas for improvement for my pipeline include tuning and improving the threshold functions that generate binary images and better masking. This would likely address the issues noted above.  Also, if this technique were to be developed further, dynamic sizing of the thresholds (i.e. standard deviations) would be prudent.  

Other improvements that could be made in my pipeline are related to my generation of birds eye images. I currently use discrete pixel values to set the source and destination values. Percentages would be be better, so the pipeline could handle different size images. Also, it would be worth exploring if image analysis could be used to identify control points on both the source and destination images. 


<strong>Checklist</strong>
* <del>Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.</del>
* <del>Apply a distortion correction to raw images.</del>
* <del>Use color transforms, gradients, etc., to create a thresholded binary image.</del>
* <del>Apply a perspective transform to rectify binary image ("birds-eye view").</del>
* <del>Detect lane pixels and fit to find the lane boundary.</del>
* <del>Determine the curvature of the lane and vehicle position with respect to center.</del>
* <del>Warp the detected lane boundaries back onto the original image.</del>
* <del>Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.</del>
