"""



"""
import os
import sys
#import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Cropping2D, Lambda, Conv2D, Flatten, BatchNormalization, Activation, ELU
from keras.regularizers import l2

import matplotlib
if not "DISPLAY" in os.environ:
	matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import ipdb

import data_server

#def Nvidia
def nvidia_model(in_shape):
	#model = Sequential()
	# https://stackoverflow.com/questions/41925765/keras-cropping2d-changes-color-channel
	# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
	# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	#input_img = Input(shape=in_shape)
	model = Sequential()

	model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=in_shape))
	model.add(Lambda(lambda x: x / 255.0 - 0.5))

	#model.add(Conv2D(3, (5, 5), activation='relu'))

	model.add(Conv2D(24, (5, 5)))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))

	model.add(Conv2D(36, (5, 5)))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))

	model.add(Conv2D(48, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))

	model.add(Conv2D(64, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))

	model.add(Conv2D(64, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))

	model.add(Flatten())
	#model.add(Dense(1164, activation='relu', W_regularizer=l2(1e-3)))
	model.add(Dense(100, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(50, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(10, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(1))

	print(model.summary())

	model.compile(loss="mse", optimizer="adam")
	params = {}
	params["EPOCHS"] = 30
	params["BATCH_SIZE"] = 64

	return model, params


def test_model(in_shape, show=False):
	#model = Sequential()
	# https://stackoverflow.com/questions/41925765/keras-cropping2d-changes-color-channel
	#input_img = Input(shape=in_shape)
	model = Sequential()

	model.add(Cropping2D(cropping=((50, 25), (0,0)), input_shape=in_shape))
	model.add(Lambda(lambda x: x / 255.0 - 0.5))

	# colorspace transform
	model.add(Conv2D(3, (1, 1), activation='relu'))

	model.add(Conv2D(8, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation(activation='elu'))

	model.add(Conv2D(24, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation(activation='elu'))

	model.add(Conv2D(36, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation(activation='elu'))

	model.add(Flatten())
	model.add(Dense(64, activation='elu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(32, activation='elu', kernel_regularizer=l2(1e-3)))

	model.add(Dense(1))

	model.compile(loss="mse", optimizer="adam")

	params = {}
	params["EPOCHS"] = 10
	params["BATCH_SIZE"] = 32
	return model, params

def generate_model(model_name, in_shape):
	print(model_name)
	if model_name == "test":
		model, params = test_model(in_shape)
	elif model_name == "nvidia":
		model, params = nvidia_model(in_shape)
	else:
		raise Exception("select test/nvidia models")
	return model, params

if __name__ == "__main__":
	model_name = "nvidia"
	if len(sys.argv) > 1:
		model_name = sys.argv[-1]

	model, params = generate_model(model_name=model_name, in_shape=(160, 320, 3))
	EPOCHS = params["EPOCHS"]
	BATCH_SIZE = params["BATCH_SIZE"]

	# train_generator = data_server.batch_generator(train_type='train', batch_size=BATCH_SIZE)
	# validation_generator  = data_server.batch_generator(train_type='valid', batch_size=BATCH_SIZE)
	# for batch_x, batch_y in train_generator:
	# 	in_shape = batch_x[0].shape
	# 	break
	# train_generator = data_server.batch_generator(train_type='train', batch_size=BATCH_SIZE)
	# imshow_cropped(batch_x[0])
	# ipdb.set_trace()
	train_generator = data_server.DataGenerator("train", batch_size=BATCH_SIZE, shuffle=True)
	valid_generator = data_server.DataGenerator("valid", batch_size=BATCH_SIZE, shuffle=True)
	# model.fit_generator(generator(features, labels, batch_size), samples_per_epoch=50, nb_epoch=10)
	# samples_per_epoch = data_server.Process().samples_per_epoch(batch_size=BATCH_SIZE)
	validation_steps = np.ceil(data_server.Process().total_samples("valid") / BATCH_SIZE)
	if not os.path.exists("model.h5") or not os.path.exists("model_weights.h5"):
		model.fit_generator(
			generator=train_generator,
			verbose=1,
			validation_data=valid_generator,
			validation_steps=validation_steps,
			epochs=EPOCHS)
			#		max_queue_size=10,
			#		workers=3,
			#		use_multiprocessing=True)

		model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
		model.save_weights('model_weights.h5')


	model.load_weights('model_weights.h5', by_name=True)
	if not os.path.exists("model.h5"):
		model.save('model.h5')  # creates a HDF5 file 'my_model.h5'


