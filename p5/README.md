**Vehicle Detection Project**

This repo contains my submission for the <strong>Vehicle Detection Project</strong>, and this readme describes my approach and the application components. All of the code that defines and trains my classifier, as well as defines and executes my pipeline, is found in app.py. Much of the logic used for loading and processing data was deligated to the utils.py file. The subsequent sections outline how these components fulfill the requirements defined in the project <a href='https://review.udacity.com/#!/rubrics/513/view' target='_blank'>rubric</a>.

[//]: # (Image References)
[point1]: output_images/car_not_car.png "training data sample"

[point2]: output_images/point2.png "Undistorted Image"
[point3]: output_images/point3.png "Perspective Transform"
[point4]: output_images/point4.png "Threshold/Binary image"
[point5]: output_images/point5.png "Histogram"
[point5_1]: output_images/point5_1.png "Sliding Window"
[point6]: output_images/point6.png "Final Result"
[bad_result]: output_images/bad_result.png "Bad results"


###Histogram of Oriented Gradients (HOG)
####1. Data
My first step was to choose and load training my data. Ultimately, I chose to only use the KITTI vision benchmark suite for vehicle data in addition to the the entire non-vehicle dataset. More discussion of my data choice can be found in the following sections, and the code to load my training data can be found on lines 24-25 in app.py. 
<pre>
VEHICLE_IMAGES = 'data/vehicles/**/*.png'<br>
NON_VEHICLE_IMAGES = 'data/non-vehicles/**/*.png'<br>
<br>
#load data<br>
vehicle_data = utils.load_data(VEHICLE_IMAGES)<br>
non_vehicle_data = utils.load_data(NON_VEHICLE_IMAGES<br>
</pre>

<strong>Sample of vehicle and non-vehicle training data</strong>
<img src='https://github.com/joshpierro/self-driving-car/blob/master/p5/output_images/car_not_car.png'>

####2. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extract the training data from the training images in lines 31-53 in app.py.

<pre>
vehicle_features = utils.extract_features(vehicle_data,
                                          color_space=utils.COLORSPACE,
                                          spatial_size=utils.SPATIAL_SIZE,
                                          hist_bins=utils.HIST_BINS,
                                          orient=utils.ORIENT,
                                          pix_per_cell=utils.PX_PER_CELL,
                                          cell_per_block=utils.CELL_PER_BLOCK,
                                          hog_channel=utils.HOG_CHANNEL,
                                          spatial_feat=utils.SPATIAL_FEATURES,
                                          hist_feat=utils.HIST_FEATURES,
                                          hog_feat=utils.HOG_FEATURES)

non_vehicle_features = utils.extract_features(non_vehicle_data,
                                          color_space=utils.COLORSPACE,
                                          spatial_size=utils.SPATIAL_SIZE,
                                          hist_bins=utils.HIST_BINS,
                                          orient=utils.ORIENT,
                                          pix_per_cell=utils.PX_PER_CELL,
                                          cell_per_block=utils.CELL_PER_BLOCK,
                                          hog_channel=utils.HOG_CHANNEL,
                                          spatial_feat=utils.SPATIAL_FEATURES,
                                          hist_feat=utils.HIST_FEATURES,
                                          hog_feat=utils.HOG_FEATURES)
</pre>

This calls the function 'extract_features' which can be found on lines 164-214 in the utils script. This function, as well as most others in this exercise were taken from the lessons.   

To get a feel for training image HOGs, I played around a bit with the 'get_hog_features' function (lines 44-61) in the utils.py script. My main focus at this point was, however, to get the feature extraction working and I didn't really tune my parameters until I my classifier was trained and I started making predictions. The image below shows a training image and its HOG, with my final parameters, which I discuss in the next section. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p5/output_images/hog_exploration.png'>


####3. Explain how you settled on your final choice of HOG parameters.
I explored many different combinations of colorspaces and HOG parameters. In the end, my final configuration was based on two things; accuracy of my classifier and the final performance on the test images, test video and project video. I ended using The following parameters:
<pre>
COLORSPACE = 'YUV' <br>
ORIENT = 9 <br>
PX_PER_CELL = 8 <br>
CELL_PER_BLOCK = 2 <br>
HOG_CHANNEL = 0 #Y <br>
</pre>

####4. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I chose a linear SVM based on the advice in the lectures, and used HOG, spatial and histogram features. I also made sure I split the traing/test data (80%/20%) before training. Once my paramters were reigned in, I was able to consitently get around 98% accuracy in my predictions with my test data. This feature can be found in lines 71-101 in app.py. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p5/output_images/training.png'>


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

<strong>Checklist</strong>
* <del>Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier</del>
* <del>Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. </del>
* <del>Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.</del>
* <del>Implement a sliding-window technique and use your trained classifier to search for vehicles in images.</del>
* <del>Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.</del>
* <del>Estimate a bounding box for vehicles detected.</del>


