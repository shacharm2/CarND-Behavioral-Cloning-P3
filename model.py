import data_server
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Cropping2D, Lambda, Conv2D, Flatten
from keras import backend
from matplotlib import pyplot as plt
import numpy as np
import ipdb

# data_generator = data_server.generate()

def imshow_cropped(image):

	model = Sequential()
	model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=image.shape))

	cropped_output = backend.function([model.layers[0].input], [model.layers[0].output])
	new_image = cropped_output([image[None,...]])[0]

	plt.imshow(new_image[0,...]/255, cmap='gray')
	plt.show()


def generate_model(in_shape, show=False):
	#model = Sequential()
	# https://stackoverflow.com/questions/41925765/keras-cropping2d-changes-color-channel
	#input_img = Input(shape=in_shape)
	model = Sequential()

	model.add(Cropping2D(cropping=((50, 25), (0,0)), input_shape=in_shape))
	model.add(Lambda(lambda x: x / 255.0 - 0.5))
	model.add(Conv2D(3, (3, 3), activation='relu'))
	model.add(Conv2D(24, (3, 3), activation='relu'))
	# model.add(Conv2D(36, (3, 3), activation='relu'))	
	# model.add(Conv2D(64, (3, 3), activation='relu'))	
	model.add(Flatten())
	# model.add(Dense(512))
	model.add(Dense(128))
	model.add(Dense(32))
	model.add(Dense(1))

	model.compile(loss="mse", optimizer="adam")
	return model

	






	return model

if __name__ == "__main__":

	BATCH_SIZE = 32
	if False:
		_valid_generator = data_server.batch_generator(train_type='valid', batch_size=BATCH_SIZE)
		for i, (batch_x, batch_y) in enumerate(_valid_generator):
			pass		
		print("valid", i)

	train_generator = data_server.batch_generator(train_type='train', batch_size=BATCH_SIZE)
	validation_generator  = data_server.batch_generator(train_type='valid', batch_size=BATCH_SIZE)
	for batch_x, batch_y in train_generator:
		in_shape = batch_x[0].shape
		break
	train_generator = data_server.batch_generator(train_type='train', batch_size=BATCH_SIZE)
	# imshow_cropped(batch_x[0])
	# ipdb.set_trace()

	EPOCHS=1
	# model.fit_generator(generator(features, labels, batch_size), samples_per_epoch=50, nb_epoch=10)
	samples_per_epoch = data_server.Process().samples_per_epoch(batch_size=BATCH_SIZE)
	validation_steps = np.ceil(data_server.Process().total_samples("valid") / BATCH_SIZE)
	model = generate_model(in_shape)
	model.fit_generator(
		generator=train_generator,
		samples_per_epoch=samples_per_epoch,
		verbose=1,
		validation_data=validation_generator,
		validation_steps=validation_steps)

	ipdb.set_trace()
