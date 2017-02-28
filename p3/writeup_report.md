#**Behavioral Cloning** 

This repo contains my submission for the <strong> Behavioral Cloning </strong> project. This readme describes its components, model architecture, training approach and other findings. 

####1. The following critical files have been included in this project:

* model.py 
* utils.py 
* drive.py 
* model.h5
* writeup_report.md 

####2. The simulator can be run in autonomous mode by executing the following script (with no arguments) 

```sh
python drive.py
```
I did my work in the pycharm IDE, so pycharm can be used to launch the drive.py script as well. Simply pull the repo, open the simulator in autonomous mode and execute drive.py. 


####3. Model/Pipeline - Utils functions

My model.py file contains code that loads training data, defines my convolution neural network, trains and validates the network and saves it. Methods in the utils.py script were used to delegate much of the logic for loading and transforming data. 


###Model Architecture and Training Strategy

####1. Model Architecture

My final model was based on the architecture published by the autonmous vehicle team at NVDIA. In the end,  it turned out that this proven architecture was much better at solving the problem than any of the architectures that I came up with, as well as other approaches I tried, specifically the <a href='https://github.com/commaai/research'>Comma AI model</a>. 

The biggest takeway I got from this exercise is to <strong>embrace the Keras framework</strong>, as well as time tested approaches to model architecture. For example, the normalization layers and cropping layers included with Keras were far easier to implement and provided a much better result that my home grown solutions. Also, the NVDIA model performed far superior to anything I came up with. 


####2. Overfitting Approach
In the end I used four dropout layers, all set to 25%, in my model to prevent overfitting. One after my convolutional layers, and one in between each connected layer. An enormous amount of trial and error performed to find this optimal configuration. 

In earlier attempts, I tried to cull out certain percent of images with 0 angles, as well as angles that were close to zero. As I was removing these images randomly, my results were consistently inconsistent. Dropout in the model turned out to be far superior. Again, another case to embrace the framework. 

####3. Hyperparameters/Tuning

As a result of lack of data, bad data and poor model architecture, I spent quite a bit of time fine tuning my model's hyper parameters. The following paramters allowed me to get the car around the track:

* Adam optimizer,with a learning rate of .0001.
* Mean Squared error loss
* a batch size of 100 
* 7 Epochs 

This is not a parameter, but worth mentioning. When creating and tuning my network architecture, I reduced my image sizes by 4x. This helped me iterate faster and give me a general idea how the model would do when larger images were used. I was able to get the car around the track with 1/2 sized images.   

####4. Training Data 

Getting the right data for this project proved to be one of the biggest challenges. I tried to collect my own data, use the Udacity data set with augmented data, and at one point use a hybrid of Udacity and collected data. In the end, I used the Udacity data with augmentations. More detail about the specific augmentations can be found below.  

###Model Architecture and Training Strategy

####1. Solution Design Approach

My high level strategy for completing this project was to get it working end to end then iterate until I got the car driving around the track (get it done, get it right, get it good).  I started with the udacity dataset and a simple sequential CNN based in the Keras lab. Once I was able to train a network and get the simulator working in autonomous mode I focused on the data. 

####2. Traing Data & Training Process
I began by trying to collect my own traing data, but since I didn't have a joystick and I was never able to collect enough quality data on my own to get anything that performed satisfactory. So, I went down the path of using udacity data. 

At first, I attempted to manually omit images that had a high frequency (zero and near zero angles) in an effort to combat over fitting and improve the training data. This approach was utimately replaced by the use of dropout layers, but the depricated code can still be seen in utils.py. 

Next, I focused on the data augmentations, beginning with the right and left camera angles. I added the left and right camera images to my training data, as well as a small negative (right side) and positive (left side) value to the steering angle to compensate for the image position. Many thanks blog posts and confluence threads that covered this topic (participate in the forums!!) 

After I had my left and right images/angles added, I copied each image and applied a random brightness value to that image and added them to the training data. 

Finally, I made a copy of each image that had a non zero angle, horizontally flipped it and then multiplied its steering angle by -1. 

I experimented with image jittering and shifting, but was never able to improve the experience with these techniques, only slow down my model compilation time. 

All of my augmentations were done on the fly inside of a python generator. At first, I resited the idea of using a generator, but I fell in love with the technique once I implemented it. Resizing, transforming and augmenting data on the fly improved my processing speed by orders of magnitude! Handy libraries like sklearn, mathlab, numpy and cv also helped alleviate the tedious tasks of data wrangling, shuffling training data and processing image data. The mathlab plotting functions were also useful for visualizing MSE loss in training and validation data. 

####3. Final Model Architecture

Again, my final architecture was based on the proven NVDIA architecture. And, a proper architecture proved to be more important than anything else. I started off with an architecture of my own design and experimented with the comma AI architecture. Once I implemented NVDIA however, my car was able to drive farther than ever, even with a minimal amount of training data! 

The final architecture includes a lambda layer that normalizes the data, a Cropping2D layer that masks removes unnecessary parts of an image (i.e. sky,hood), five convolutional layers to filter the data followed by four fully connected layers. The table below provides a visual summary overview. 

<pre>
Layer (type)                     Output Shape          Param '#     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 80, 160, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 33, 160, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 29, 156, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 25, 152, 36)   21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 11, 74, 48)    43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 9, 72, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 7, 70, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 7, 70, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 31360)         0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           3136100     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_4[0][0]                  
====================================================================================================
Total params: 3,273,019
Trainable params: 3,273,019
Non-trainable params: 0
____________________________________________________________________________________________________
None
</pre>

