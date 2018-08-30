""" This module generates csv_metadata

	resources:
		https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
		https://arxiv.org/pdf/1710.05381.pdf
		http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html

"""


import os
from io import BytesIO
import threading

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Cropping2D, Lambda, Conv2D, Flatten
from PIL import Image
import matplotlib
if not "DISPLAY" in os.environ:
	matplotlib.use('Agg')
from matplotlib import pyplot as plt
from shutil import copyfile
from time import time

import ipdb
print(matplotlib.get_backend())
np.random.seed(int(time()))


#https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/

PARAMS = {"crop": (50, 25)}

class threadsafe_iter(object):
	"""Takes an iterator/generator and makes it thread-safe by
		serializing call to the `next` method of given iterator/generator.abs

		https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()


def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g


class Singleton(type):
	""" https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
	"""
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]

saved_once = False
def preprocess(image):
	# HSV/LAB/YUV
	global saved_once
	img = cv2.GaussianBlur(np.array(image), (3,3), 0)
	output = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	if not saved_once:
		fig = plt.figure()
		plt.imshow(output / 255, cmap='gray')
		plt.savefig("output_images/preprocessed_{}.png".format(np.random.randint(1000)))
		plt.close(fig)
		saved_once = True

	return output

class Process(object, metaclass=Singleton):

	def __init__(self, data_folder='/opt/carnd_p3/', shuffle=True, train_size=0.8):
		self.data_folder = data_folder
		self.folders = []
		#self.draw_one = {'flip': False, 'shear': False}
		self.out_folder = "output_images"

		submetadata = []
		filter_folders = []
		for sub_folder in os.listdir(self.data_folder):
			curr_dir = os.path.join(self.data_folder, sub_folder)
			csv_metadata = os.path.join(curr_dir, 'driving_log.csv')
			if not os.path.isdir(curr_dir) or not os.path.exists(csv_metadata):
				continue
			elif filter_folders and not sub_folder in filter_folders:
				continue

			print("Adding", curr_dir)
			self.folders.append(curr_dir)

			fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20,20))
			metadata_i = pd.read_csv(csv_metadata)

			# metadata_i['flip'] = False
			# metadata_i.loc[metadata_i['steering'].abs() < 0.01, 'flip'] = True
			to_save = []
			side_camera_bias = .25
			alpha = {'left': side_camera_bias, 'center': 0, 'right': -side_camera_bias}
			for ndir, direction in enumerate(sorted(alpha)):  #['center', 'left', 'right']:
				abs_paths = metadata_i[direction].apply(lambda subdir: os.path.join(curr_dir, subdir.strip(' ')))
				steering_angle = (metadata_i['steering'] + alpha[direction])
				to_save.append((abs_paths.iloc[0], steering_angle.iloc[0]))
				steering_angle.plot(linewidth=2, color='b', ax=axes[ndir], label='raw', alpha=0.5)
				axes[ndir].fill_between(steering_angle.index, 0, steering_angle, color='b', alpha=0.5)

				steering_angle = steering_angle.rolling(3, center=True).mean().fillna(method='ffill').fillna(method='bfill')

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
		self.metadata.loc[: , 'type'] = 'train'  # will be splitted later

		# augment only non zero steering angles - abundant & (angle == -angle ) is redundant		
		self.metadata.loc[:, 'identity'] = True
		self.metadata.loc[:, 'flip'] = False
		self.metadata.loc[:, 'half'] = False
		self.metadata.loc[:, 'shear'] = False
		self.metadata.loc[:, 'translate'] = False

		# show images
		fig, axes = plt.subplots(nrows=1, ncols=len(to_save), figsize=(20,20))
		for i, (img_filename, al) in enumerate(to_save):
			image = self.open(img_filename, preprocess_flag=False)

			p0 = np.array((image.shape[0], image.shape[1] / 2))

			dy = 0.2 * image.shape[0] # pixels
			dx = dy * np.tan(al)
			p1 = p0 + np.array((dx, -dy))
			X = np.array([p1[0], p0[0]])
			Y = np.array([p1[1], p0[1]])
			axes[i].imshow(image / 255, cmap='gray')
			axes[i].plot(X, Y, linewidth=10)
		else:
			plt.savefig("output_images/1_three_views_{}.png".format(sub_folder), bbox_inches='tight')
		plt.close(fig)

		# reduce 0 angle samples
		fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,20))
		self.metadata['steering'].hist(bins=int(np.sqrt(len(self.metadata))), ax=axes[0], label='raw')
		self.metadata['steering'].plot.density(bw_method='scott', ax=axes[1], label='raw')
		self.max_train_angle = self.metadata['steering'].abs().max()

		# how to choose the fraction
		# max(np.histogram(self.metadata.loc[~filt, "steering"], bins=int(np.sqrt(len(self.metadata))))[0])
		frac = 0.1
		if False:
			nonzero_df = self.metadata[self.metadata['steering'] != 0]
			zero_df = self.metadata[self.metadata['steering'] == 0].sample(frac=frac)
		elif True:
			# filt = self.metadata['steering'].abs().isin([0, side_camera_bias])
			filt = (self.metadata['steering'].abs() < 0.01)

			# perhaps these shouldn't be removed
			filt |= ((self.metadata['steering'] - side_camera_bias).abs() < 0.01)
			filt |= ((self.metadata['steering'] + side_camera_bias).abs() < 0.01)

			nonzero_df = self.metadata[~filt]
			zero_df = self.metadata[filt].sample(frac=frac)
			#n_in = int(len(zero_df) * frac)
			#print("n_in=", n_in, len(zero_df))
			#zero_df = zero_df.iloc[:n_in]
			#translate_md = zero_df.iloc[n_in:]
			#translate_md = translate_md.sample(frac=0.)
		else:
			filt = self.metadata['steering'].abs().eq(0)

			# perhaps these shouldn't be removed
			filt |= self.metadata['steering'].abs().eq(side_camera_bias)

			nonzero_df = self.metadata[~filt]
			zero_df = self.metadata[filt].sample(frac=frac)


		self.metadata = pd.concat([zero_df, nonzero_df], axis='rows')
		self.split(shuffle, train_size)

		#train_filt = (self.metadata.loc[train_idx, 'train_type'] == 'train')
		train_filt = self.metadata['train_type'].eq('train')

		## augmentations ##
		# oversample > 25 deg
		#(self.metadata['steering'] > 24.5 * np.pi / 180)
		oversample = True
		if oversample:
			filt_large_angle = self.metadata['steering'].gt(24.5 * np.pi / 180)
			filt_large_angle |= self.metadata['steering'].lt(-20 * np.pi / 180)  #< -20 * np.pi / 180)
			dups = self.metadata[filt_large_angle]# & train_filt]
			self.metadata = pd.concat([self.metadata, dups], axis='rows', ignore_index=True)


		self.metadata['steering'].hist(bins=int(np.sqrt(len(self.metadata))), color='r', alpha=0.5, ax=axes[0], label='preprocessing')
		self.metadata['steering'].plot.density(bw_method='scott', ax=axes[1], color='r', alpha=0.5, label='preprocessing')
		axes[0].legend()
		axes[1].legend()
		plt.savefig("output_images/1_steering_histogram.png".format(sub_folder))
		plt.close(fig)


		# lazy augmentations

		#filt = train_filt & (self.metadata['steering'].abs() < 0.01)
		#filt = train_filt & self.metadata['steering'].abs().lt(0.01)
		filt = self.metadata['steering'].abs().gt(0.01)
		filt |= ((self.metadata['steering'] - side_camera_bias).abs() < 0.01)
		filt |= ((self.metadata['steering'] + side_camera_bias).abs() < 0.01)

		#translate_md.loc[:, 'identity'] = False
		#translate_md.loc[:, 'translate'] = True

		translate_md = self.metadata[filt].sample(frac=0.3)
		translate_md.loc[:, 'identity'] = False
		translate_md.loc[:, 'translate'] = True

		flip_md = self.metadata[filt].sample(frac=1)
		flip_md.loc[:, 'identity'] = False
		flip_md.loc[:, 'flip'] = True

		half_md = self.metadata[filt].sample(frac=1)
		half_md.loc[:, 'identity'] = False
		half_md.loc[:, 'half'] = True

		shear_md = self.metadata[filt].sample(frac=0.3)
		shear_md.loc[:, 'identity'] = False
		shear_md.loc[:, 'shear'] = True

		# concat all augmentations
		self.metadata = pd.concat([self.metadata, flip_md, shear_md, translate_md, half_md], axis='rows', ignore_index=True, sort=False).sample(frac=1)

		# save 
		for _ in range(10):
			random_image = np.random.randint(len(self.metadata))

			al = self.metadata.iloc[random_image]['steering']
			full_image_name = self.metadata.iloc[random_image]['image']
			image_name = os.path.splitext(os.path.split(full_image_name)[-1])[0]
			#image = np.asarray(Image.open(full_image_name))
			image = self.open(full_image_name, preprocess_flag=False)

			p0 = np.array((image.shape[0], image.shape[1] / 2))
			dy = 0.2 * image.shape[0] # pixels
			dx = dy * np.tan(al)	# tan(al) = dx/dy 
			p1 = p0 + np.array((dx, -dy))

			self.imshow_augmentations(image, image_name=image_name, velocity=(p0, p1, al), save=True)
		else:
			image = self.open(self.metadata.iloc[0]['image'])

	def split(self, shuffle=True, train_size=0.7):
		""" train/valid/test split """
		if shuffle:
			self.shuffle() # = self.metadata.sample(frac=1)

		if 'type' not in self.metadata:
			self.metadata.loc[: ,'type'] = None
		train_idx, test_val_idx = train_test_split(self.metadata.index, train_size=train_size)
		val_idx, test_idx = train_test_split(test_val_idx, train_size=0.9)

		self.metadata.loc[train_idx, 'train_type'] = 'train'
		self.metadata.loc[val_idx, 'train_type'] = 'valid'
		self.metadata.loc[test_idx, 'train_type'] = 'test'
		self.dump_metadata()

	def dump_metadata(self):
		self.metadata.to_hdf("metadata.hdf", key="metadata")

	def load_metadata(self):
		self.metadata = pd.read_hdf("metadata.hdf", key="metadata")

	def imshow_augmentations(self, image, image_name=None, velocity=None, save=False, show=False):
		fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,20))
		model = Sequential()
		model.add(Cropping2D(cropping=(PARAMS['crop'], (0,0)), input_shape=image.shape))

		cropped_output = keras.backend.function([model.layers[0].input], [model.layers[0].output])
		new_image = cropped_output([image[None,...]])[0]
		for iimg in [0, 2, 3]:
			axes[iimg][0].imshow(image / 255, cmap='gray')
			axes[iimg][0].set_title('raw')

		axes[0][1].imshow(new_image[0, ...] / 255, cmap='gray')
		axes[0][1].set_title('cropped')

		if velocity is not None:
			p0, p1, al = velocity
			X = np.array([p1[0], p0[0]])
			Y = np.array([p1[1], p0[1]])
			for iimg in [0, 2, 3]:
				axes[iimg][0].plot(X, Y, linewidth=10)
				axes[iimg][0].set_title("steering {0:2.1f}".format(al * 180 / np.pi))

			# show half image obscure
			half_image, steering = self.mask_half(image, velocity[2])
			axes[3][1].imshow(half_image / 255, cmap='gray')
			axes[3][1].plot(X, Y, linewidth=10)
			axes[3][1].set_title("Occlusion, steering {0:2.1f}".format(steering * 180 / np.pi))

			# flip
			flipped, flip_al = self.flip(image, al)
			p0 = np.array((image.shape[0], image.shape[1] / 2))
			dy = 0.2 * image.shape[0]
			dx = dy * np.tan(flip_al)
			p1 = p0 + np.array((dx, -dy))
			X = np.array([p1[0], p0[0]])
			Y = np.array([p1[1], p0[1]])

			axes[2][1].imshow(flipped / 255, cmap='gray')
			axes[2][1].plot(X, Y, linewidth=10)
			axes[2][1].set_title("steering {0:2.1f}".format(flip_al * 180 / np.pi))

			

		#axes[0][1].imshow(new_image[0, ...] / 255, cmap='gray')
		#axes[0][1].set_title('cropped')

		#if velocity is not None:
			augmentations = [self.shear, self.translate]
			title = ["sheared cropped", "translated cropped"]
			for i in range(1): #range(len(augmentations)):
				augmented, augment_angle = augmentations[i](image, velocity[2], self.max_train_angle)
				axes[i+1][0].imshow(augmented, cmap='gray')

				p0 = np.array((image.shape[0], image.shape[1] / 2))
				dy = 0.2 * image.shape[0]
				dx = dy * np.tan(augment_angle)
				p1 = p0 + np.array((dx, -dy))
				X = np.array([p1[0], p0[0]])
				Y = np.array([p1[1], p0[1]])
				axes[i+1][0].plot(X, Y, linewidth=10)
				axes[i+1][0].set_title("steering {0:2.1f}".format(augment_angle * 180 / np.pi))

				model = Sequential()
				model.add(Cropping2D(cropping=(PARAMS['crop'], (0,0)), input_shape=image.shape))

				cropped_output = keras.backend.function([model.layers[0].input], [model.layers[0].output])
				augmented_cropped = cropped_output([augmented[None,...]])[0]
				axes[i+1][1].imshow(augmented_cropped[0, ...] / 255, cmap='gray')
				axes[i+1][1].set_title(title[i])


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
		if metadata['identity']:
			return image, metadata['steering']

		# (2) flip
		elif metadata['flip']:
			return self.flip(image, metadata['steering'])

		# (3) shear
		elif metadata['shear']:
			return self.shear(image, metadata['steering'], self.max_train_angle)

		# (4) shear
		elif metadata['translate']:
			return self.translate(image, metadata['steering'], self.max_train_angle)

		# (5) half mask
		elif metadata['half']:
			return self.mask_half(image, metadata['steering'])

		else:
			raise Exception("how did you get here?")
		# TODO: rotate ..
		# TODO: exagerate opposite angle and camera

	@staticmethod
	def open(image_file, preprocess_flag=True):

		with Image.open(image_file) as fd:
			img = np.asarray(fd)# Image.open(image_file)
		return img
		#if not preprocess_flag:
		#	return img
		#return preprocess(img)
		img = cv2.GaussianBlur(img, (3,3), 0)
		return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

	@staticmethod
	def flip(image, steering_angle):
		return cv2.flip(image, 1), -steering_angle   # cv2.flip(image, 1) or np.fliplr(image)

	@staticmethod
	def shear(image, steering_angle, max_angle):
		#angle_range = sorted([steering_angle, max_angle * np.sign(steering_angle)])
		abs_angle = abs(steering_angle)
		a = np.random.uniform(abs_angle, max_angle)
		#a = min(a, max_angle)
		steering_angle_out = a * np.sign(steering_angle)

		rows, cols = image.shape[:2]
		# around middle point
		dx = 0.5 * rows * np.tan(steering_angle_out)
		shifted = [0.5 * cols + dx, 0.5 * rows]
		pt_in = np.float32([[0, rows],[cols, rows], [cols / 2, rows / 2]])
		pt_out = np.float32([[0, rows],[cols, rows], shifted])
		M = cv2.getAffineTransform(pt_in, pt_out)
		image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

		return image, steering_angle_out

	@staticmethod
	def mask_half(image, steering_angle):
		img = image.copy()
		rows, cols = img.shape[:2]

		if steering_angle > 0:
			contours = np.array([[0, 0], [0, rows], [cols/2, rows], [cols/2 + rows * np.tan(steering_angle), 0]], np.int32)
		else:
			contours = np.array([[cols, 0], [cols, rows], [cols/2, rows], [cols/2 + rows * np.tan(steering_angle), 0]], np.int32)

		cv2.fillPoly(img, pts=[contours], color=(0, 0, 0))


		#if np.random.rand() < 0.5:
		#	img[:, :cols//2] = 0
		#else:
		#	img[:, cols//2:] = 0

		return img, steering_angle

	@staticmethod
	def translate(image, steering_angle, max_angle):
		steering_angle_out = steering_angle
		while steering_angle_out == steering_angle:
			abs_angle = abs(steering_angle)
			a = np.random.uniform(abs_angle, max_angle)
			steering_angle_out = a * np.sign(steering_angle)

		rows, cols = image.shape[:2]
		dx = 0.5 * rows * np.tan(steering_angle_out)
		shifted = [0.5 * cols + dx, 0.5 * rows]
		pt_in = np.float32([[0, rows], [cols / 2, rows / 2], [cols + dx, rows]])
		pt_out = np.float32([[dx, rows], [cols / 2 + dx, rows / 2], [cols + dx, rows]])
		M = cv2.getAffineTransform(pt_in, pt_out)
		image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

		return image, steering_angle_out


	def get_submetadata(self, train_type, index, batch_size):
		""" """
		filt = self.metadata['train_type'].eq(train_type)
		submetadata = self.metadata[filt].iloc[index * batch_size:(index+1) * batch_size]
		return submetadata

	def get(self, train_type):
		""" """
		for _, row in self.metadata[self.metadata['train_type'] == train_type].iterrows():
			#img = np.asarray(Image.open(row['image']))
			img = self.open(row['image'])
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
			#img = np.asarray(Image.open(row['image']))
			img = self.open(row['image'])

			if X is None or y is None:
				X = np.empty((batch_size, *img.shape))
				y = np.empty((batch_size), dtype=int)

			# augment must yield a single image. 
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
		metadata = Process().get_submetadata(self.train_type, index, self.batch_size)

		# Generate data
		#X, y = Process().data_generation(indices, self.batch_size)
		#data_generation(self, indices, batch_size, height, width, nchannels):

		X = None
		y = None
		# Generate data
		for j, (i, row) in enumerate(metadata.iterrows()):

			#img = np.asarray(Image.open(row['image']))
			#img = self.open(row['image'])
			with Image.open(row['image']) as fd:
				img = np.asarray(fd)# Image.open(image_file)
			if X is None or y is None:
				X = np.empty((self.batch_size, *img.shape))
				y = np.empty((self.batch_size), dtype=float)

			# augment must yield a single image. 
			# for _x, _y in Process().augment(img, row):
			# 	X[j,] = _x
			# 	y[j] = _y
			X[j,], y[j] = Process().augment(img, row)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			Process().shuffle()


if __name__ == '__main__':

	BATCH_SIZE = 64
	data_gen = DataGenerator("train", batch_size=BATCH_SIZE, shuffle=True)
	print(data_gen[1])
	print(len(data_gen))

	Process().metadata
	ipdb.set_trace()
