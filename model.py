import csv
import cv2
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import ndimage

# Parameters
epochs = 5
validation_split = 0.2
correction = 0.2
row, col, ch = 160, 320, 3
batch_size = 32

# Read in each row/line from driving_log.csv
lines = [] # samples
with open('training_data_own/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path ='training_data_own/IMG/' + filename
                    image = ndimage.imread(current_path)
                    images.append(image)

                # create adjusted steering measurements for the side camera images
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # add angles to data set
                measurements.extend([steering_center])
                measurements.extend([steering_left])
                measurements.extend([steering_right])
            
            # Data augmentation
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement)
                augmented_measurements.append(measurement*-1.0)

            # Convert images and steering measurements to numpy arrays since that's the format Keras requires
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

# Using Generators
train_samples, validation_samples = train_test_split(lines, test_size=validation_split)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Setup Keras
from keras.models import Sequential # class provides common functions like fit(), evaluate() and compile()
from keras.models import Model
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Build the Neural Network
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3))) 
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Conv2D(24, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(36, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
model.add(Conv2D(64, (1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split = validation_split, shuffle=True, = epochs) # default number of epochs is 10 in Keras
history_object = model.fit_generator(train_generator, steps_per_epoch = ceil(len(train_samples)/batch_size), validation_data = validation_generator, validation_steps = ceil(len(validation_samples)/batch_size), epochs=epochs, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss (instead printing, save it when using command line terminal)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show() # Only use this in Jupyter Notebooks. Do not use this in the command line terminal which can't display any graphics. Instead save the file (see next line)
plt.savefig('examples/loss_visualization.png')