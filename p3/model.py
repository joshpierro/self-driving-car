#imports
import utils
import json
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D,Cropping2D,MaxPooling2D
from keras.layers.core import Lambda, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Flatten

import tensorflow as tf
tf.python.control_flow_ops = tf

#CONSTANTS
DRIVING_DATA_PATH = 'data/driving_log.csv'
CHECKPOINT = 'checkpoints/model{epoch:02d}.h5'
OPTIMIZER = Adam(lr=.0001) #.0001 1e-5
LOSS = 'mse'
BATCH_SIZE = 100
EPOCHS = 8

#load the data
print('loading data')
culled_list = utils.load_data(DRIVING_DATA_PATH)
train_samples, validation_samples = train_test_split(culled_list, test_size=0.2)
print("number of culled rows for training: " , len(train_samples))
print("number of culled rows for validation: " , len(validation_samples))

#NVDIA model

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='tanh'))
model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=['accuracy'])

#generator
def generate(samples, batch_size):
    num_samples = len(samples)
    shuffle(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size][:]
            images = []
            angles = []

            for batch_sample in batch_samples:
                #TODO decompose this stuff into functions
                #load center
                img_src = 'data/' + batch_sample[0].strip()
                center_image = mpimg.imread(img_src)
                center_image = utils.resize_image(center_image)
                images.append(center_image)
                angles.append(str(batch_sample[3]))

                #add random brightness
                center_bright = utils.bright_image(center_image)
                images.append(center_bright)
                angles.append(str(batch_sample[3]))

                # flip center
                if float(batch_sample[3]) != 0:
                    flipped_image = utils.flip_image(center_image)
                    flipped_angle = utils.flip_angle(float(batch_sample[3]))
                    images.append(flipped_image)
                    angles.append(flipped_angle)

                ####################################################

                # left
                img_src = 'data/' + batch_sample[1].strip()
                left_image = mpimg.imread(img_src)
                left_image = utils.resize_image(left_image)
                images.append(left_image)
                left_angle = float(batch_sample[3]) + .28
                angles.append(str(left_angle))

                left_bright = utils.bright_image(left_image)
                images.append(left_bright)
                angles.append(left_angle)

                #flip left
                if float(batch_sample[3]) != 0:
                    left_flipped_image = utils.flip_image(center_image)
                    left_flipped_angle = utils.flip_angle(float(batch_sample[3])) + .28
                    images.append(left_flipped_image)
                    angles.append(left_flipped_angle)


                ################################################

                #right
                img_src = 'data/' + batch_sample[2].strip()
                right_image = mpimg.imread(img_src)
                right_image = utils.resize_image(right_image)
                images.append(right_image)
                right_angle = float(batch_sample[3]) - .28
                angles.append(str(right_angle))

                right_bright = utils.bright_image(right_image)
                images.append(right_bright)
                angles.append(right_angle)

                #flip right
                if float(batch_sample[3]) != 0:
                    right_flipped_image = utils.flip_image(center_image)
                    right_flipped_angle = utils.flip_angle(float(batch_sample[3])) - .28
                    images.append(right_flipped_image)
                    angles.append(right_flipped_angle)

                x_train = np.array(images)
                y_train = np.array(angles)

            yield shuffle(x_train, y_train)


#save model checkpoints
checkpoint = ModelCheckpoint(CHECKPOINT,
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto')

# generate data
training = generate(train_samples,BATCH_SIZE)
validation = generate(validation_samples,BATCH_SIZE)

history = model.fit_generator(training,
                        samples_per_epoch=len(train_samples),
                        nb_epoch=EPOCHS,
                        verbose=1,
                        callbacks=[checkpoint], #uncomment this if you want to save checkpoints
                        validation_data=validation,
                        nb_val_samples=len(validation_samples))
print('trained')

#save the model
model.save('model.h5')
json_string = model.to_json()
json.dump(json_string, open('model.json', 'w'))
print('model saved')

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show(block=True)

