import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import generator
from sklearn.model_selection import train_test_split
from scipy import ndimage
from PIL import Image

# Parameters
epochs = 5
validation_split = 0.2
correction = 0.2
row, col, ch = 160, 320, 3  # Trimmed image format
batch_size = 32

# Read in each row/line from log-file into lines[]
lines =[] # samples
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = [] # steering angles between -1.0 and 1.0
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path ='/opt/carnd_p3/data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        
    # create adjusted steering measurements for the side camera images
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # add angles to data set
    measurements.extend([steering_center])
    measurements.extend([steering_left])
    measurements.extend([steering_right])

"""
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path ='/opt/carnd_p3/data/IMG' + filename
    # image = cv2.imread(current_path)
    # images.append(image)  
        
    # create adjusted steering measurements for the side camera images
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    path = "/opt/carnd_p3/data/IMG/" # path to training IMG directory
    img_center = np.asarray(Image.open(path + line[0].split('/')[-1]))
    # print('image shape: ', img_center.shape) # DEBUG
    img_left = np.asarray(Image.open(path + line[1].split('/')[-1]))
    img_right = np.asarray(Image.open(path + line[2].split('/')[-1]))

    # add images and angles to data set
    images.extend(img_center)
    measurements.extend([steering_center])
    images.extend(img_left)
    measurements.extend([steering_left])
    images.extend(img_right)
    measurements.extend([steering_right]) 
"""
    
print('shape of first image "images[0]: ', images[0].shape) # DEBUG
    
# Data augmentation
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement*-1.0)
    
# Convert images and steering measurements to numpy arrays since that's the format Keras requires
X_train = np.array(augmented_images)
print('X_train shape: ', X_train.shape) # DEBUG
y_train = np.array(augmented_measurements)

# Setup Keras
from keras.models import Sequential # class provides common functions like fit(), evaluate() and compile()
from keras.models import Model
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Build the Neural Network
model = Sequential()

# Preprocessing the data: Normalizing and mean-centering
# first layer: lambda layer for simple operations can be used to create arbitrary functions that operate on each image as it passes through the layer. Here, lambda layer will ensure that the model will normalize input images when making predictions
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (row, col, ch))) # provide here the input_shape. In Keras only needed for the first layer. Following will automatically infered from previous layers.
model.add(Cropping2D(cropping=((75,25),(0,0)))) # crop images (75 pixels from upper end, 25 from lower end and 0 from the sides); also try (50,20)
model.add(Conv2D(24, (5,5), strides=(2, 2), padding='valid')) # syntax for Keras 2.0.9
model.add(Conv2D(36, (5,5), strides=(2, 2), padding='valid'))
model.add(Conv2D(48, (5,5), strides=(2, 2), padding='valid'))
model.add(Conv2D(64, (3,3), strides=(2, 2), padding='valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
    
model.summary()

model.compile(loss='mse', optimizer='adam')

# Using Generators
train_samples, validation_samples = train_test_split(lines, test_size=validation_split)

# compile and train the model using the generator function
train_generator = generator.generator(train_samples, batch_size=batch_size)
validation_generator = generator.generator(validation_samples, batch_size=batch_size)

model.fit(X_train, y_train, validation_split = validation_split, shuffle=True, epochs = epochs) # default number of epochs is 10 in Keras
model.save('model.h5')

history_object = model.fit_generator(train_generator, samples_per_epoch = ceil(len(train_samples)/batch_size), validation_data = validation_generator, nb_val_samples = ceil(len(validation_samples)/batch_size), nb_epoch = epochs, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()