"""



"""
import os
import sys
#import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Cropping2D, Lambda, Conv2D, Flatten, BatchNormalization, Activation, ELU, Dropout, MaxPooling2D, merge
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras import metrics

from keras.regularizers import l2

import matplotlib
if not "DISPLAY" in os.environ:
	matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import ipdb

import data_server
import trace
trace.trace_start("trace.html")
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
	params["EPOCHS"] = 10
	params["BATCH_SIZE"] = 64

	return model, params

def test_model(in_shape, show=False):
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 1.0, input_shape=in_shape))
	model.add(Cropping2D(cropping=(data_server.PARAMS['crop'], (0,0)), input_shape=in_shape))

	model.add(Conv2D(3, (1, 1), activation='relu'))

	for i in range(2):
		model.add(Conv2D(8, (3, 3)))
		model.add(BatchNormalization())
		model.add(Activation(activation='relu'))

		model.add(Conv2D(8, (3, 3)))
		model.add(BatchNormalization())
		model.add(Activation(activation='relu'))

		model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Flatten())
	model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(16, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(1))
	print(model.summary())


	model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3), metrics=[metrics.mean_squared_error])

	params = {}
	params["EPOCHS"] = 10
	params["BATCH_SIZE"] = 128

	return model, params


class DenseNet(Model):
	# https://towardsdatascience.com/densenet-2810936aeebb
	def __init__(self, in_shape=None, show=False, inputs=None, outputs=None, name=None):
				#super().__init__(inputs=inputs, outputs=x)
		if inputs and outputs and name:
			super().__init__(inputs=inputs, outputs=outputs, name=name)
			return


		channels = 8
		self.stage = 0
		self.set_params()
		#uodel = Sequential()
		inputs = Input(shape=in_shape, name="input")
		x = Lambda(lambda x: x/255 - 1.0)(inputs)
		#model.add(Cropping2D(cropping=(data_server.PARAMS['crop'], (0,0)), input_shape=in_shape))

		# dense block
		for i in range(4):
			x_block = self.conv_block(x, channels)
			x = concatenate([x, x_block])

		x = self.bottleneck_block(x, channels)


		# dense block
		for i in range(4):
			x_block = self.conv_block(x, channels)
			x = concatenate([x, x_block])

		x = self.bottleneck_block(x, channels)


		# output
		x = Flatten()(x)
		x = Dense(64, activation='relu', kernel_regularizer=l2(1e-3), name="FC_stage_{}".format(self.stage))(x)
		x = Dense(32, activation='relu', kernel_regularizer=l2(1e-3))(x)
		x = Dense(16, activation='relu', kernel_regularizer=l2(1e-3))(x)
		x = Dense(1)(x)
		super().__init__(inputs=inputs, outputs=x, name=name)

		print(self.summary())
		self.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3), metrics=[metrics.mean_squared_error])

	def set_params(self):
		self.params = {}
		self.params["EPOCHS"] = 10
		self.params["BATCH_SIZE"] = 64


	def bottleneck_block(self, x, channels):
		method = 'bottleneck_block'
		x = Conv2D(channels, (1, 1), padding='same', name="{}_conv_stage_{}".format(method, self.stage))(x)
		x = MaxPooling2D((3, 3), strides=(2, 2), name='{}_pool_stage_{}'.format(method,self.stage))(x)

		self.stage += 1
		return x

	def conv_block(self, x, channels):
		method = 'conv_block'
		x = Conv2D(channels, (1, 1), padding='same', name="{}_conv_1x1_stage_{}".format(method, self.stage))(x)
		x = BatchNormalization(axis=3, name="{}_BN_1x1_stage_{}".format(method, self.stage))(x)
		x = Activation(activation='relu', name="{}_relu_1x1_stage_{}".format(method, self.stage))(x)

		x = Conv2D(channels, (3, 3), padding='same', name="{}_conv_3x3_stage_{}".format(method, self.stage))(x)
		x = BatchNormalization(axis=3, name="{}_BN_3x3_stage_{}".format(method, self.stage))(x)
		x = Activation(activation='relu', name="{}_relu_3x3_stage_{}".format(method, self.stage))(x)
		self.stage += 1

		return x


	#def dense_block(self):
	#	#https://github.com/flyyufelix/DenseNet-Keras/blob/master/densenet121.py


def inception_model(in_shape, show=False):
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 1.0, input_shape=in_shape))
	model.add(Cropping2D(cropping=(data_server.PARAMS['crop'], (0,0)), input_shape=in_shape))

	model.add(Conv2D(3, (1, 1), activation='relu'))

	for i in range(2):
		model.add(Conv2D(8, (3, 3)))
		model.add(BatchNormalization())
		model.add(Activation(activation='relu'))

		model.add(Conv2D(8, (3, 3)))
		model.add(BatchNormalization())
		model.add(Activation(activation='relu'))

		model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Flatten())
	model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(16, activation='relu', kernel_regularizer=l2(1e-3)))
	model.add(Dense(1))
	print(model.summary())


	model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3), metrics=[metrics.mean_squared_error])

	params = {}
	params["EPOCHS"] = 10
	params["BATCH_SIZE"] = 128
	return model, params

def generate_model(model_name, in_shape):
	print(model_name)
	if model_name == "test":
		model, params = test_model(in_shape)
	elif model_name == "nvidia":
		model, params = nvidia_model(in_shape)
	elif model_name.lower() == "densenet":
		model = DenseNet(in_shape)
		params = model.params
	else:
		raise Exception("select test/nvidia models")
	return model, params

def main():
	model_name = "nvidia"
	if len(sys.argv) > 1:
		model_name = sys.argv[-1]

	model, params = generate_model(model_name=model_name, in_shape=(160, 320, 3))
	if not os.path.exists("model.h5") and os.path.exists('model_weights.h5'):
		model.load_weights('model_weights.h5', by_name=True)
		model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
		return


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

	prev = True
	if prev:
		train_generator = data_server.DataGenerator("train", batch_size=BATCH_SIZE, shuffle=True)
		valid_generator = data_server.DataGenerator("valid", batch_size=BATCH_SIZE, shuffle=True)
	else:
		train_generator = data_server.batch_generator("train", batch_size=BATCH_SIZE)
		valid_generator = data_server.batch_generator("valid", batch_size=BATCH_SIZE)

	#train_generator = data_server.batch_generator(train_type='train', batch_size=None):
	# model.fit_generator(generator(features, labels, batch_size), samples_per_epoch=50, nb_epoch=10)
	#samples_per_epoch = data_server.Process().samples_per_epoch(batch_size=BATCH_SIZE)
	validation_steps = data_server.Process().total_samples("valid") // BATCH_SIZE
	train_steps = data_server.Process().total_samples("terain") // BATCH_SIZE
	if not os.path.exists("model.h5") or not os.path.exists("model_weights.h5"):
		if prev:
			model.fit_generator(
				use_multiprocessing=False,
				workers=1,
				generator=train_generator,
				verbose=1,
				validation_data=valid_generator,
				epochs=EPOCHS)
		else:
			model.fit_generator(
				use_multiprocessing=False,
				generator=train_generator,
				samples_per_epoch=train_steps, #samples_per_epoch // BATCH_SIZE,
				verbose=1,
				validation_data=valid_generator,
				validation_steps=validation_steps,
				epochs=EPOCHS)



		model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
		model.save_weights('model_weights.h5')


	#data_server.Process().load_metadata()
	#metadata = data_server.Process().metadata


if __name__ == "__main__":
	DenseNet((160, 32, 3))
	main()

