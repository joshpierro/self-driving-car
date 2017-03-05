**Advanced Lane Finding Project**

This repo contains my submission for the <strong>Advanced Lane Finding</strong> project, and this readme describes my approach and its components. The subsequent sections outline how these components fulfill the requirements defined in the <a href='https://review.udacity.com/#!/rubrics/571/view'>project rubric</a>

My app.py file contains all of the code that defines and execute my pipeline. Methods in the utils.py script were used to delegate much of the logic used to process the images.

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
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

