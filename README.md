# Behavioral Cloning for Autonomous Vehicles

## Overview
This repository contains files for **cloning a human driver's behavior** and training an autonomous vehicle to imitate that behaviour.

Deep Learning methods and a convolutional neural network are used to clone the driving behavior. With input from camera data, the network will output steering angles for an autonomous vehicle. The model is built, trained, validated and tested using Keras.

For data collection, a car can be steered around a track in a simulator. Collected image data and steering angles are used to train the neural network. It was trained and validated on different data sets to ensure that the model is not overfitting. Then this model is used to drive the car autonomously around the track - again in the simulator. Here is the result:

[![IMAGE ALT TEXT HERE](./examples/Thumbnail.png)](https://youtu.be/IfGLIvO4KnY)

## The Project
The steps were the following:
* Use the simulator to drive the car around and collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around in the simulator

## Details About Files In This Repository
### `drive.py`

Usage of `drive.py` requires a trained convolution neural network to be saved as an h5 file `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### `model.py`

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

The model was not built from scratch. The following, existing NVIDIA pipeline for autonomous driving served as a starting point.

![alt text][image9]

Feeding the NVIDIA model with 160x320 sized images (RGB which means 3 channels) from the training set (recorded with the camera in the simulator) 'model.summary()' returns the following information about the shapes of the network layers.

![alt text][image10]

Training the model with this architecture was not satisfactory. The vehicle was able to drive autonomously and very centrally through the course, but it shaked and fidgeted a lot. That would mean little comfort for the passengers.

Many iterative steps later after modifying the model, adding new layers, testing, removing layers again, testing different parameters, fixing bugs, etc., the convolution neural network finally still consists of 5 convolutional layers and 4 dense layers (just like the NVIDIA example) but with serveral other layer types on top. Here is the summary:

![alt text][image11]

Data is being normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Dropout layers where added. Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. 

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually.

#### 4. Creation of training data

In the simulator, you can weave all over the road and turn recording on and off to record recovery driving (I will explain this later). In a real car, however, that’s not really possible bacuase it is not legally. So in a real car, we’ll have multiple cameras on the vehicle, and we’ll map recovery paths from each camera. For example, if you train the model to associate a given image from the center camera with a left turn, then you could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn. And you could train the model to associate the corresponding image from the right camera with an even harder left turn. In that way, you can simulate your vehicle being in different positions, somewhat further off the center line. For that purpose, the simulator captures images from three cameras mounted on the car: a left, center and right camera:

![alt text][image13]

Training data was generated to keep the vehicle driving on the road. I used a combination of center lane driving, recovering driving and driving counterclockwise. The latter is additionally realized by data augmentation (flipping images vertically and multiplying angles by -1.0 accordingly). Here you can see an example:

![alt text][image14]

To capture good driving behavior, I first recorded two laps on track driving in the center.

I then recorded the vehicle recovering from the sides (left or right) to the center so that the vehicle would learn how to get back to the center in case it strays from the middle while driving autonomously.

Finally I recorded two laps driving in the center counterclockwise.

#### 5. Creation of the Training Set & Training Process

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over- or underfitting. A good number of epochs turned out to be 5.

#### 6. Loss visualization

![alt text][image12]

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/Thumbnail.png "Thumbnail"
[image9]: ./examples/NVIDIA_architecture.png "NVIDIA Architecture"
[image10]: ./examples/model_architecture_old.png "NVIDIA Summary"
[image11]: ./examples/model_architecture.png "Final architecture"
[image12]: ./examples/loss_visualization.png "Loss visualization"
[image13]: ./examples/multiple_cameras.png "Multiple cameras"
[image14]: ./examples/Data_augmentation.png "Data augmentation"
