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
I did my work in a pycharm project, so that can be used to launch the drive.py script as well. 


####3. Model/Pipeline - Utils functions

My model.py file contains code that loads training data, defines my convolution neural network, trains and validates the network and saves it. Methods in the utils.py script were used to delegate much of the logic for loading and transforming data. 


###Model Architecture and Training Strategy

####1. Model Architecture

My final model was based on the architecture published by the autonmous vehicle team at NVDIA. In the end,  it turned out that this proven architecture was much better at solving the problem than any of the architectures that I came up with, as well as other approaches I tried, specifically the <a href='https://github.com/commaai/research'>Comma AI model</a>. 

The network includes a lambda layer that normalizes the data, a Cropping2D layer that masks removes unnecessary parts of an image (i.e. sky,hood), five convolutional layers to filter the data followed by four fully connected layers. The table below provides a visual summary overview. 


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

Getting the right data for this project proved to be one of the biggest challenges. I tried to collect my own data, use the Udacity data set with augmented data, as well as a hybrid of Udacity and collected data. I didn't have a joystick and I was never able to collect enough quality data on my own to get anything that performed satisfactory. So, I ended up using augmented Udacity data. 

More detail about the specific augmentations can be found below.  

###Model Architecture and Training Strategy

####1. Solution Design Approach



The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
