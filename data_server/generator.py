""" This module generates csv_metadata

	resources:
		https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
		https://arxiv.org/pdf/1710.05381.pdf

"""


import os
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Cropping2D, Lambda, Conv2D, Flatten
from PIL import Image
import matplotlib
print(matplotlib.get_backend())
if not "DISPLAY" in os.environ:
	matplotlib.use('Agg')
from matplotlib import pyplot as plt
import ipdb



class Singleton(type):
	""" https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
	"""
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]

class Process(object, metaclass=Singleton):

	def __init__(self, data_folder='/opt/carnd_p3/', shuffle=True, train_size=0.7):
		self.data_folder = data_folder
		self.folders = []

		self.out_folder = "output_images"

		submetadata = []
		for sub_folder in os.listdir(self.data_folder):
			curr_dir = os.path.join(self.data_folder, sub_folder)
			if not os.path.isdir(curr_dir):
				continue

			self.folders.append(curr_dir)
			csv_metadata = os.path.join(curr_dir, 'driving_log.csv')

			fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20,20))

			metadata_i = pd.read_csv(csv_metadata)

			# metadata_i['flip'] = False
			# metadata_i.loc[metadata_i['steering'].abs() < 0.01, 'flip'] = True
			

			side_camera_bias = .25
			alpha = {'left': side_camera_bias, 'center': 0, 'right': -side_camera_bias}
			for ndir, direction in enumerate(sorted(alpha)):  #['center', 'left', 'right']:
				abs_paths = metadata_i[direction].apply(lambda subdir: os.path.join(curr_dir, subdir.strip(' ')))
				steering_angle = (metadata_i['steering'] + alpha[direction])
				steering_angle.plot(linewidth=2, color='b', ax=axes[ndir], label='raw', alpha=0.5)
				axes[ndir].fill_between(steering_angle.index, 0, steering_angle, color='b', alpha=0.5)

				steering_angle = steering_angle.rolling(4, center=True).mean().fillna(method='ffill').fillna(method='bfill')
				
				steering_angle.plot(linewidth=2, color='tab:olive', ax=axes[ndir], label='interp', alpha=0.8)
				axes[ndir].set_title(direction)
				axes[ndir].legend()

				concat = pd.concat([abs_paths, steering_angle], axis='columns')
				concat.columns = ['image', 'steering']
				submetadata.append(concat)
			else:
				plt.savefig("output_images/1_steerings_{}.png".format(sub_folder))
				# plt.show()
		plt.close(fig)

		self.metadata = pd.concat(submetadata, ignore_index=True, sort=False)
		# reduce 0 angle samples

		fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,20))
		self.metadata['steering'].hist(bins=int(np.sqrt(len(self.metadata))), ax=axes[0], label='raw')
		self.metadata['steering'].plot.density(bw_method='scott', ax=axes[1], label='raw')
		

		frac = 0.025
		if False:
			nonzero_df = self.metadata[self.metadata['steering'] != 0]
			zero_df = self.metadata[self.metadata['steering'] == 0].sample(frac=frac)
		else:
			# filt = self.metadata['steering'].abs().isin([0, side_camera_bias])
			filt = (self.metadata['steering'].abs() < 0.01)
			filt |= ((self.metadata['steering'] - side_camera_bias).abs() < 0.01)
			filt |= ((self.metadata['steering'] + side_camera_bias).abs() < 0.01)

			nonzero_df = self.metadata[~filt]
			zero_df = self.metadata[filt].sample(frac=frac)

		# how to choose the fraction
		# max(np.histogram(self.metadata.loc[~filt, "steering"], bins=int(np.sqrt(len(self.metadata))))[0])

		self.metadata = pd.concat([zero_df, nonzero_df], axis='rows')

		# oversample > 25 deg
		filt_large_angle = (self.metadata['steering'] > 24.5 * np.pi / 180)
		filt_large_angle |= (self.metadata['steering'] < -20 * np.pi / 180)
		dups = self.metadata[filt_large_angle]
		self.metadata = pd.concat([self.metadata, dups, dups], axis='rows', ignore_index=True)

		# augment only non zero steering angles - abundant & (angle == -angle ) is redundant		
		self.metadata.loc[:, 'flip'] = False
		filt = (self.metadata['steering'].abs() < 0.01)
		filt |= ((self.metadata['steering'] - side_camera_bias).abs() < 0.01)
		filt |= ((self.metadata['steering'] + side_camera_bias).abs() < 0.01)

		flip_md = self.metadata[filt]
		flip_md.loc[:, 'flip'] = True
		self.metadata = pd.concat([self.metadata, flip_md], axis='rows', ignore_index=True)

		self.metadata['steering'].hist(bins=int(np.sqrt(len(self.metadata))), color='r', alpha=0.5, ax=axes[0], label='preprocessing')
		self.metadata['steering'].plot.density(bw_method='scott', ax=axes[1], color='r', alpha=0.5, label='preprocessing')
		axes[0].legend()
		axes[1].legend()
		plt.savefig("output_images/1_steering_histogram.png".format(sub_folder))

		plt.close(fig)
		if shuffle:
			self.shuffle() # = self.metadata.sample(frac=1)

		self.metadata.loc[: ,'type'] = None
		train_idx, test_val_idx = train_test_split(self.metadata.index, train_size=train_size)
		val_idx, test_idx = train_test_split(test_val_idx, train_size=0.5)

		self.metadata.loc[train_idx, 'train_type'] = 'train'
		self.metadata.loc[val_idx, 'train_type'] = 'valid'
		self.metadata.loc[test_idx, 'train_type'] = 'test'

		# save 
		for _ in range(10):
			random_image = np.random.randint(len(self.metadata))

			al = self.metadata.loc[random_image, 'steering']
			full_image_name = self.metadata.loc[random_image, 'image']
			image_name = os.path.splitext(os.path.split(full_image_name)[-1])[0]
			image = np.asarray(Image.open(full_image_name))
			
			p0 = np.array((image.shape[0], image.shape[1] / 2))
			dy = 0.2 * image.shape[0] # pixels
			# tan(al) = dx/dy 
			dx = dy * np.tan(al)
			p1 = p0 + np.array((dx, -dy))

			self.imshow_cropped(image, image_name=image_name, velocity=(p0, p1, al), save=True)

	@staticmethod
	def imshow_cropped(image, image_name=None, velocity=None, save=False, show=False):
		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,20))
		model = Sequential()
		model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=image.shape))

		cropped_output = keras.backend.function([model.layers[0].input], [model.layers[0].output])
		new_image = cropped_output([image[None,...]])[0]
		axes[0].imshow(image / 255, cmap='gray')
		if velocity is not None:
			p0, p1, al = velocity
			X = np.array([p1[0], p0[0]])
			Y = np.array([p1[1], p0[1]])
			# U = np.sin(X)
			# V = np.cos(Y)
			# axes[0].quiver(X, Y, U, V)
			# ipdb.set_trace()
			
			axes[0].plot(X, Y, linewidth=10)
			axes[0].set_title("steeing {0:2.1f}".format(al * 180 / np.pi))

		axes[1].imshow(new_image[0, ...] / 255, cmap='gray')

		if show:
			plt.show()
		
		if save and image_name is None:
			plt.savefig('output_images/1_cropped.png', bbox_inches='tight')
		elif save:
			plt.savefig('output_images/1_cropped_{}.png'.format(image_name), bbox_inches='tight')
			

		plt.close(fig)

	def shuffle(self):
		""" """
		self.metadata = self.metadata.sample(frac=1)
		self.metadata = self.metadata.sample(frac=1)
		self.metadata = self.metadata.sample(frac=1)

	def samples_per_epoch(self, batch_size, train_type='train'):
		""" """
		filt = (self.metadata['train_type'] == train_type)
		total_train_images = len(self.metadata[filt])
		samples_per_epoch = total_train_images - total_train_images % batch_size
		return samples_per_epoch

	def total_samples(self, train_type):
		""" """
		filt = (self.metadata['train_type'] == train_type)
		return len(self.metadata[filt])

	def augment(self, image, metadata):
		""" """
		# for xi in range(1):
		# 	yield image, steering
		#if row['augment'] == 'flip':

		# (1) identity
		yield image, metadata['steering']

		# (2) flip
		if metadata['flip']:
			# or .. flipped = cv2.flip(image, 1)
			flipped = np.fliplr(image)
			steering = - metadata['steering']
			yield flipped, steering

		# TODO: rotate ..
		# TODO: sheer ..
		# TODO: exagerate opposite angle and camera

	def get_indices(self, train_type, index, batch_size):
		""" """
		filt = (self.metadata['train_type'] == train_type)
		indices = self.metadata[filt].iloc[index * batch_size:(index+1) * batch_size].index
		return indices

	def get(self, train_type):
		""" """
		for _, row in self.metadata[self.metadata['train_type'] == train_type].iterrows():
			img = np.asarray(Image.open(row['image']))
			for _x, _y in self.augment(img, row):
				yield _x, _y

	def data_generation(self, indices, batch_size):
		""" """
		# batch_size *= # of augmentations!
		batch_size *= 1
		X = None #np.empty((batch_size, height, width, nchannels))
		y = None #np.empty((batch_size), dtype=int)

		# Generate data
		for i, index in enumerate(indices):
			row = self.metadata.loc[index]
			img = np.asarray(Image.open(row['image']))

			if X is None or y is None:
				X = np.empty((batch_size, *img.shape))
				y = np.empty((batch_size), dtype=int)

			for _x, _y in self.augment(img, row):
				X[i,] = _x
				y[i] = _y

		return X, y
		# for i, (ID) in enumerate(list_IDs_temp):
		# 	# Store sample
		# 	X[i,] = np.load('data/' + ID + '.npy')

		# 	# Store class
		# 	y[i] = self.labels[ID]

		# return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



def batch_generator(train_type='train', batch_size=None):
	""" """
	assert batch_size is not None
	process = Process()
	batch_x, batch_y = None, None
	i_batch = 0
	while True:
		_x, _y = next(process.get(train_type))
		if batch_x is None:
			height, width, nchannels = _x.shape
			batch_x = np.zeros((batch_size, height, width, nchannels), np.uint8)
			batch_y = np.zeros(batch_size)

		batch_x[i_batch % batch_size] = _x
		batch_y[i_batch % batch_size] = _y
		if i_batch % batch_size == batch_size - 1:
			yield batch_x, batch_y
			batch_x, batch_y = None, None
		
		i_batch += 1
		print(len(batch_x))


	# for i_batch, (_x, _y) in enumerate(process.get(train_type)):
	# 	if batch_x is None:
	# 		height, width, nchannels = _x.shape
	# 		batch_x = np.zeros((batch_size, height, width, nchannels), np.uint8)
	# 		batch_y = np.zeros(batch_size)

	# 	batch_x[i_batch % batch_size] = _x
	# 	batch_y[i_batch % batch_size] = _y
	# 	if i_batch % batch_size == batch_size - 1:
	# 		yield batch_x, batch_y
	# 		batch_x, batch_y = None, None

class DataGenerator(keras.utils.Sequence):
	# def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), nchannels=1,
	#              n_classes=10, shuffle=True):
	# def __init__(self, train_type, batch_size, height, width, nchannels, shuffle=True):

	def __init__(self, train_type, batch_size, shuffle=True):
		self.train_type = train_type
		# self.height = height
		# self.width = width
		self.batch_size = batch_size
		# self.nchannels = nchannels
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		#return int(np.floor(len(self.list_IDs) / self.batch_size))
		return int(np.floor(Process().total_samples(self.train_type) / self.batch_size))

	def __getitem__(self, index):
		indices = Process().get_indices(self.train_type, index, self.batch_size)
		
		# Generate data
		X, y = Process().data_generation(indices, self.batch_size)
		#data_generation(self, indices, batch_size, height, width, nchannels):

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			Process().shuffle()


if __name__ == '__main__':

	BATCH_SIZE = 128
	data_gen = DataGenerator("train", batch_size=BATCH_SIZE, shuffle=True)
	print(data_gen[1])
	print(len(data_gen))

	Process().metadata
	ipdb.set_trace()
