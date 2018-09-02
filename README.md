# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

PROJECT SPECIFICATION
---
| Topic  | CRITERIA   | MEETS SPECIFICATIONS |
|---|---|---|
|Required Files | Are all required files submitted? | The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.   |
| Quality of Code  | Is the code functional?  | The model provided can be used to successfully operate the simulation.  |
| Quality of Code  | Is the code usable and readable?  | The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed. |
| Model Architecture and Training Strategy  | Has an appropriate model architecture been employed for the task? |The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.  | 
| Model Architecture and Training Strategy  | Has an attempt been made to reduce overfitting of the model?  | Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.  |
| Model Architecture and Training Strategy  | Have the model parameters been tuned appropriately?  |Learning rate parameters are chosen with explanation, or an Adam optimizer is used.   |
| Model Architecture and Training Strategy  | Is the training data chosen appropriately?  | Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track). |
| Architecture and Training Documentation  | Is the solution design documented?  | The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.   |
| Architecture and Training Documentation  | Is the model architecture documented?  | The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.  |
| Architecture and Training Documentation  | Is the creation of the training dataset and training process documented? |  The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included. |
| Simulation | Is the car able to navigate correctly on test data?  | No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). |



# Project code

### Usage

### Training

    python3 model.py 

There are additional models, such as Nvidia and DenseNet that can be called as:

    python3 model.py densenet
    python3 model.py nvidia

### Autonomous mode

Run the drive sequence with the matching model:

	python3 drive.py test

## Code details

(1) data_server/generator.py

The data generator inherits from Keras' Sequence class, designed for data serving. In order to use the class, two methods, namely, __len__ & __getitem__ needed to be implemented:

    class DataGenerator(keras.utils.Sequence):
		def __len__(self):
			...

		def __getitem__(self, index):
			...

model.py

Straight forward usage - build the model (Final model uses the Sequential API), compile and serve through Keras' fit_generator() API

trace.py

Used for multithreading debuging


# Model Architecture and Training Strategy

The goal was building a model that sucessfully drives continuously, while reducing model size as much as possible.

The code uses a straight forward deep architecture with repeating blocks. 

Each block is composed of:

1. 2D Convolution, 3 channels, 1x1 convolution, ReLU activation

2. 2D Convolution block

	2.1. 2D Convolution - 8 channels, 3x3 convolution

	2.2. Batch normalization

	2.3. Relu activation

3. 2D Convolution block

	3.1. 2D Convolution - 8 channels, 3x3 convolution

	3.2. Batch normalization

	3.3. Relu activation

4. Max pooling 2D, 2x2 strides

This block is repeated twice