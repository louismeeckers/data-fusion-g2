import os

import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
import albumentations as A

DATA_PATH = '../data/'

class PlantDataset(torch.utils.data.Dataset):
	def __init__(self, set_dir, transform=None, target_transform=None):
		self.set_dir = set_dir

		self.plant_directories = []
		self.client_directories = ['A1', 'A2', 'A4', 'A6', 'B1', 'B2', 'B3', 'B4']
		self.labels = pd.read_csv(os.path.join(DATA_PATH, 'seedling_labels.csv'))
		self.labels['Label'] = self.labels.apply(lambda x: ((((x['Expert 1'] + x['Expert 2'] + x['Expert 3'] + x['Expert 4']) / 4) - 1) / 3), axis=1)

		set_df_labels = []
		for client_dir in self.client_directories:
			client_labels = self.labels[self.labels['Rfid'] == client_dir]
			if self.set_dir == 'train':
				set_df_labels.append(client_labels[:100])
			elif self.set_dir == 'valid':
				set_df_labels.append(client_labels[100:])

		self.set_labels = pd.concat(set_df_labels)

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.set_labels)

	def __getitem__(self, index):
		item_label = self.set_labels.iloc[index]

		# Images
		image_color = Image.open(os.path.join(DATA_PATH, item_label['color_cam_path'])).convert('RGB')
		image_color = np.array(image_color, dtype=np.float32) # / 255.0

		image_side = Image.open(os.path.join(DATA_PATH, item_label['side_cam_path'])).convert('RGB')
		image_side = np.array(image_side, dtype=np.float32) # / 255.0
		
		if self.transform is not None:
			image_color = self.transform(image=image_color)['image']
			image_side = self.transform(image=image_side)['image']

		image_color = torch.tensor(np.transpose(image_color, (2, 0, 1))) # tensor (3, 960, 1280)
		image_side = torch.tensor(np.transpose(image_side, (2, 0, 1)))
		
		# Label
		label = item_label['Label']

		return image_color, image_side, label