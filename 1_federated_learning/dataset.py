import os

import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
import albumentations as A

DATA_PATH = '../data/'

class PlantDataset(torch.utils.data.Dataset):
	def __init__(self, client_dir, transform=None, target_transform=None):
		self.client_dir = client_dir

		self.plant_directories = sorted(os.listdir(os.path.join(DATA_PATH, client_dir)))
		self.labels = pd.read_csv(os.path.join(DATA_PATH, 'seedling_labels.csv'))  

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.plant_directories)

	def __getitem__(self, index):
		plant_directory =  self.plant_directories[index]
		plant_id = plant_directory[6:16]
		plant_labels = self.labels[(self.labels['Rfid'] == self.client_dir) & (self.labels['Pos'] == plant_id)].iloc[0]

		# Images
		image_color = Image.open(os.path.join(DATA_PATH, plant_labels['color_cam_path'])).convert('RGB')
		image_color = np.array(image_color, dtype=np.float32) / 255.0

		image_side = Image.open(os.path.join(DATA_PATH, plant_labels['side_cam_path'])).convert('RGB')
		image_side = np.array(image_side, dtype=np.float32) / 255.0
		
		# Label
		label = ((plant_labels['Expert 1'] + plant_labels['Expert 2'] + plant_labels['Expert 3'] + plant_labels['Expert 4']) / 4) - 1 # [0, 3]
		# label = 1 if label > 1.5 else 0
		label /= 3

		if self.transform is not None:
			image_color = self.transform(image=image_color)['image']
			image_side = self.transform(image=image_side)['image']

		image_color = torch.tensor(np.transpose(image_color, (2, 0, 1))) # tensor (3, 960, 1280)
		image_side = torch.tensor(np.transpose(image_side, (2, 0, 1)))

		return image_color, image_side, label