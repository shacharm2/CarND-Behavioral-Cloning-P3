""" This module generates data """


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from io import BytesIO
import ipdb
import numpy as np
import base64

class Singleton(type):
	""" https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
	"""
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]

class Process(object, metaclass=Singleton):

	def __init__(self, data_folder='/opt/behavioral_cloning', shuffle=True, train_size=0.7):
		self.data_folder = data_folder
		self.folders = []

		metadata = pd.DataFrame(columns=['image', 'steering', 'type'])
		submetadata = []
		for sub_folder in os.listdir(self.data_folder):
			curr_dir = os.path.join(self.data_folder, sub_folder)
			if not os.path.isdir(curr_dir):
				continue
			csv_metadata = os.path.join(curr_dir, 'driving_log.csv')
			
			metadata_i = pd.read_csv(csv_metadata)
			temp = pd.DataFrame(index=metadata_i.index)
			alpha = {'left': .25, 'center': 0, 'right': .25}
			for direction in ['center', 'left', 'right']:
				abs_paths = metadata_i[direction].apply(lambda subdir: os.path.join(curr_dir, subdir.strip(' ')))
				concat = pd.concat([abs_paths, metadata_i['steering'] + alpha[direction]], axis='columns')
				concat.columns = ['image', 'steering']
				submetadata.append(concat)

		self.metadata = pd.concat(submetadata, ignore_index=True, sort=False)
		if shuffle:
			self.metadata = self.metadata.sample(frac=1)

		self.metadata.loc[: ,"type"] = None
		train_idx, test_val_idx = train_test_split(self.metadata.index, train_size=train_size)
		val_idx, test_idx = train_test_split(test_val_idx, train_size=0.5)

		self.metadata.loc[train_idx, "train_type"] = "train"
		self.metadata.loc[val_idx, "train_type"] = "valid"
		self.metadata.loc[test_idx, "train_type"] = "test"

	def samples_per_epoch(self, batch_size, train_type="train"):
		filt = (self.metadata["train_type"] == train_type)
		total_train_images = len(self.metadata[filt])
		samples_per_epoch = total_train_images - total_train_images % batch_size
		return samples_per_epoch

	def total_samples(self, train_type):
		filt = (self.metadata["train_type"] == train_type)
		return len(self.metadata[filt])

	def augment(self, image, steering):
		for xi in range(1):
			yield image, steering


	def get(self, train_type):
		for index, row in self.metadata[self.metadata["train_type"] == train_type].iterrows():
			img = np.asarray(Image.open(row['image']))
			for _x, _y in self.augment(img, row['steering']):
				yield _x, _y

def batch_generator(train_type='train', batch_size=None):
	assert batch_size is not None
	process = Process()
	batch_x, batch_y = None, None
	for i_batch, (_x, _y) in enumerate(process.get(train_type)):
		if batch_x is None:
			height, width, nchannels = _x.shape
			batch_x = np.zeros((batch_size, height, width, nchannels), np.uint8)
			batch_y = np.zeros(batch_size)

		batch_x[i_batch % batch_size] = _x
		batch_y[i_batch % batch_size] = _y
		if i_batch % batch_size == batch_size - 1:
			yield batch_x, batch_y
			batch_x, batch_y = None, None


if __name__ == "__main__":

	BATCH_SIZE = 128
	_valid_generator = batch_generator(train_type='valid', batch_size=BATCH_SIZE)
	for i, (batch_x, batch_y) in enumerate(_valid_generator):
		pass		
	print("valid", i)


	_test_generator = batch_generator(train_type='test', batch_size=BATCH_SIZE)
	for i, (batch_x, batch_y) in enumerate(_test_generator):
		pass		
	print("test", i)

	_train_generator = batch_generator(train_type='train', batch_size=BATCH_SIZE)
	for i, (batch_x, batch_y) in enumerate(_train_generator):
		pass		
	print("train", i)


# generator = iter(process)
# process = Process()
# for x in process.get('train'):
# 	print(x)
