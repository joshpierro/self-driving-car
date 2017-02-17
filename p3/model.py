#imports
import json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from keras.layers import Convolution2D, ELU
from keras.layers.core import Lambda, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Flatten

import tensorflow as tf
tf.python.control_flow_ops = tf

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

#CONSTANTS
DRIVING_DATA_PATH = 'data/backup/driving_log.csv'
OPTIMIZER = Adam(lr=1e-5)
LOSS = 'mse'

#load the data
print('loading data')
X_train = []
Y_train = []

csv_data = pd.read_csv(DRIVING_DATA_PATH)
for index,row in csv_data.iterrows():
    #center images
    img_center = 'data/' + row['center'].strip()
    X_train.append(mpimg.imread(img_center))
    Y_train.append(str(row['steering']))
    # left images
    #img_center = 'data/' + row['left'].strip()
    #X_train.append(mpimg.imread(img_center))
    #Y_train.append(str(row['steering']))
    # right images
    #img_center = 'data/' + row['right'].strip()
    #X_train.append(mpimg.imread(img_center))
    #Y_train.append(str(row['steering']))

# Shuffle the data
print('shuffling data')
X_train = np.array(X_train)
X_train, y_train = shuffle(X_train, Y_train)

#one hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(Y_train)


#set up model
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(160, 320, 3),
                 output_shape=(160, 320, 3)))
model.add(Convolution2D(3, 1, 1, input_shape=(160, 320, 3),border_mode='same', name='color_conv'))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
#model.add(Dense(68))
model.add(Dense(124))
model.compile(loss=LOSS, optimizer=OPTIMIZER)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])

#train the model
model.fit(X_train,y_one_hot, nb_epoch=10, validation_split=0.2)
print('trained')

#save the model
model.save('model.h5')
json_string = model.to_json()
json.dump(json_string, open('model.json', 'w'))

print('model saved')


