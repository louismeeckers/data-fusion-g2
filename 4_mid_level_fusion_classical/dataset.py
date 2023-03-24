import os

import numpy as np
from PIL import Image
import pandas as pd
import albumentations as A

from utils import compute_label_01, get_experts_weights

DATA_PATH = '../data/'

class PlantDataset():
	def __init__(self, set_dir, transform=None, target_transform=None):
		self.set_dir = set_dir

		self.client_directories = ['A1', 'A2', 'A4', 'A6', 'B1', 'B2', 'B3', 'B4']
		self.df = pd.read_csv(os.path.join(DATA_PATH, 'seedling_labels.csv'))

		# self.df['Label'] = self.df.apply(lambda x: ((((x['Expert 1'] + x['Expert 2'] + x['Expert 3'] + x['Expert 4']) / 4) - 1) / 3), axis=1)
		experts_weights = get_experts_weights(self.df) # [weight_E1, weight_E2, weight_E3, weight_E4]
		self.df['Label'] = self.df.apply(lambda x: compute_label_01(x, experts_weights), axis=1)
		self.df['Label'] =self.df['Label'].round(0)
		set_df = []
		for client_dir in self.client_directories:
			client_labels = self.df[self.df['Rfid'] == client_dir]
			if self.set_dir == 'train':
				set_df.append(client_labels[:100])
			elif self.set_dir == 'valid':
				set_df.append(client_labels[100:])
			# elif self.set_dir == 'test':
				# set_df.append(client_labels[100:])

		self.set_df = pd.concat(set_df)

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.set_df)

	def __getitem__(self, index):
		item_row = self.set_df.iloc[index]

		# Images
		image_color = Image.open(os.path.join(DATA_PATH, item_row['color_cam_path'])).convert('RGB')
		image_color = np.array(image_color, dtype=np.float32) / 255.0 # normalize

		image_side = Image.open(os.path.join(DATA_PATH, item_row['side_cam_path'])).convert('RGB')
		image_side = np.array(image_side, dtype=np.float32) / 255.0 # normalize
		
		if self.transform is not None:
			image_color = self.transform(image=image_color)['image']
			image_side = self.transform(image=image_side)['image']

		
		# Label
		label = item_row['Label']

		return image_color, image_side, label