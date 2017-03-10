**Vehicle Detection Project**

This repo contains my submission for the <strong>Vehicle Detection Project</strong>, and this readme describes my approach and the application components. All of the code that defines and trains my classifier, as well as defines and executes my pipeline, is found in app.py. Much of the logic used for loading and processing data was delegated to the utils.py file. The subsequent sections outline how these components fulfill the requirements defined in the project <a href='https://review.udacity.com/#!/rubrics/513/view' target='_blank'>rubric</a>.

[//]: # (Image References)
[point1]: output_images/car_not_car.png "training data sample"
[point2]: output_images/hog_exploration.png "Car HOG"
[point3]: output_images/training.png "Training Results"
[point4]: output_images/window.png "Window Grid"
[point5]: output_images/pipe.png "Pipeline output"

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

I chose a linear SVM based on the advice in the lectures, and used HOG, spatial and histogram features. I also made sure I split the training/test data (80%/20%) before training. Once my parameters were reigned in, I was able to consistently get around 98% accuracy in my predictions with my test data. This feature can be found in lines 71-101 in app.py. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p5/output_images/training.png'>


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I found that my sliding window strategy was probably the most critical piece of the pipeline. My implementation was inspired by the lectures, but trial and error (and lots of failure) drove the final scales and properties of the windows. My solution consists of two layers of windows, one 64x64 px and one 128x128px. These two grids were slightly staggered on the x axis and both overlapped 90%. Lines 115-121 in app.py define the sliding windows. 

<pre>
windows_64 = utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[375, 430],<br>
                    xy_window=(64, 64), xy_overlap=(0.9, 0.9))<br>

windows_128 = utils.slide_window(image, x_start_stop=[70, None], y_start_stop=[375, 560],<br>
                    xy_window=(128, 128), xy_overlap=(0.9, 0.9))<br>

windows = windows_64 + windows_128
<pre>

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p5/output_images/windows.png'>

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the Y channel of the YUV color space, as well as spatially binned color and histograms of color when extracting my HOG features. Again, the sliding window size, overlap and placement made a tremendous difference. Finally, playing with the threshold to prevent false positives was helpful. Select images from my pipeline can be found below. 

<img src='https://github.com/joshpierro/self-driving-car/blob/master/p5/output_images/pipe.png'>

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are OK as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my final video: 
It can also be found in the root of this repo project_results.mp4

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In lines 146-155 of app.py I implement a thresholded heatmap to identify and and label vehicle positions. This was my main strategy for reducing false positives. I then constructed bounding boxes to cover the area of each blob detected. A richer strategy might include aggregating the results from adjacent frames in the video and set a threshold on that. This would certainly cull out more anomalies. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Again, I was pleasantly surprised that the classroom materials aligned so closely with this exercise and if you followed them you could get 80% of an end to end solution. That said, there were lots of stumbling blocks along the way, lots of trial and error was needed to get it working and my solution is far from perfect. 

Specifically, some of the issues I had include, but were not limited to; selecting the right training data, scaling my images and implementing the HOG extractions for the first time. 

Situations where my pipeline might fail, include, but are not limited to: night time driving, inclement weather, different driving conditions (super urban,super rural), switching lanes and objects unknown to the classifier, like a motorcycle, cyclist or cow crossing the road. 

To over come some of the issues described above, I think that more training data could help. For example, images if vehicles and non vehicles in inclement weather and night time conditions. Also, a hardened strategy (more dimensions) for creating my heatmaps (using adjacent frames) would also help. 

<strong>Checklist</strong>
* <del>Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier</del>
* <del>Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. </del>
* <del>Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.</del>
* <del>Implement a sliding-window technique and use your trained classifier to search for vehicles in images.</del>
* <del>Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.</del>
* <del>Estimate a bounding box for vehicles detected.</del>

