"""



"""

import data_server
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Cropping2D, Lambda, Conv2D, Flatten
from keras import backend
from matplotlib import pyplot as plt
import numpy as np
import ipdb

# data_generator = data_server.generate()

def imshow_cropped(image, save=False, show=False):

	model = Sequential()
	model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=image.shape))

	cropped_output = backend.function([model.layers[0].input], [model.layers[0].output])
	new_image = cropped_output([image[None,...]])[0]

	if save:
		plt.savefig('cropped.png', bbox_inches='tight')

	if show:
		plt.imshow(new_image[0,...]/255, cmap='gray')
		plt.show()


#def Nvidia

def generate_model(in_shape, show=False):
	#model = Sequential()
	# https://stackoverflow.com/questions/41925765/keras-cropping2d-changes-color-channel
	#input_img = Input(shape=in_shape)
	model = Sequential()

	model.add(Cropping2D(cropping=((50, 25), (0,0)), input_shape=in_shape))
	model.add(Lambda(lambda x: x / 255.0 - 0.5))
	model.add(Conv2D(3, (3, 3), activation='relu'))
	# model.add(Conv2D(24, (3, 3), activation='relu'))
	# # model.add(Conv2D(36, (3, 3), activation='relu'))	
	# # model.add(Conv2D(64, (3, 3), activation='relu'))	
	model.add(Flatten())
	# model.add(Dense(512))
	# model.add(Dense(128))
	model.add(Dense(32))
	model.add(Dense(1))

	model.compile(loss="mse", optimizer="adam")
	return model

if __name__ == "__main__":

	BATCH_SIZE = 128
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
	EPOCHS=3
	# model.fit_generator(generator(features, labels, batch_size), samples_per_epoch=50, nb_epoch=10)
	# samples_per_epoch = data_server.Process().samples_per_epoch(batch_size=BATCH_SIZE)
	validation_steps = np.ceil(data_server.Process().total_samples("valid") / BATCH_SIZE)
	model = generate_model(in_shape=(160, 320, 3))

	model.fit_generator(
		generator=train_generator,
		verbose=1,
		validation_data=valid_generator,
		validation_steps=validation_steps)

	model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
	model.save_weights('model_weights.h5')

	del model  # deletes the existing model

	model = load_model('model.h5')
	# model.load_weights('my_model_weights.h5', by_name=True)

	ipdb.set_trace()
