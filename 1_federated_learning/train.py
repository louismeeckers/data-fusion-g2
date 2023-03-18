import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import albumentations as A
import wandb

from model import CNN
from dataset import PlantDataset
from utils import load_checkpoint, save_checkpoint, optimizer_to

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '../data/'



class Client:
	def __init__(self, client_dir, config):
		self.client_dir = client_dir
		self.config = config

		# Data
		transform = A.Compose(
			[
				A.RandomBrightnessContrast(p=0.3),
				A.Rotate(limit=35, p=1.0),
				A.HorizontalFlip(p=0.5),
				A.RandomResizedCrop(height=self.config['image_height'], width=self.config['image_width'], scale=(0.6, 1.0), ratio=(0.75, 1.333), interpolation=1, p=0.5),
				A.Resize(height=self.config['image_height'], width=self.config['image_width']),
			],
		)

		resize = A.Compose(
			[
				A.Resize(height=self.config['image_height'], width=self.config['image_width']),
			],
		)

		self.dataset_train = PlantDataset(set_dir='train', client_dir=self.client_dir, transform=(transform if self.config['data_augmentation'] else resize))
		self.loader_train = torch.utils.data.DataLoader(dataset=self.dataset_train, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True, num_workers=2)

		self.dataset_valid = PlantDataset(set_dir='valid', client_dir=self.client_dir, transform=resize)
		self.loader_valid = torch.utils.data.DataLoader(dataset=self.dataset_valid, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True, num_workers=2)

		# Model
		self.model = CNN()

		# Criterion and Optimizer
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

	def receive_from_server(self, model_state_dict, optimizer_state_dict):
		self.model.load_state_dict(model_state_dict, strict=True)
		self.optimizer.load_state_dict(optimizer_state_dict)
		optimizer_to(self.optimizer, device)

	def train(self):
		self.model.to(device)

		example_ct = 0 # number of examples seen
		batch_ct = 0 # number of batches seen

		for epoch in tqdm(range(self.config['epochs']), position=2, desc='Epoch', leave=False, dynamic_ncols=True):
			# Train loop
			train_loss = 0
			self.model.train()
			for inputs_color, inputs_side, labels in tqdm(self.loader_train, position=3, desc='Batch train', leave=False, dynamic_ncols=True):
				
				if self.config['image_type'] == 'color':
					inputs = inputs_color
				elif self.config['image_type'] == 'side':
					inputs = inputs_side
				inputs = inputs.float().to(device)
				labels = labels.float().to(device)

				# Forward pass ➡
				preds = self.model(inputs).squeeze()
				loss = self.criterion(preds, labels)
				
				# Backward pass ⬅
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				train_loss += loss
				example_ct += len(inputs)
				batch_ct += 1

				# Report metrics every 5 batch
				if ((batch_ct + 1) % 5) == 0:
					tqdm.write(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.8f}")

		self.model.to('cpu')

	def send_to_server(self):
		# return weights of the local model
		return self.model.state_dict(), self.optimizer.state_dict()


class Server:
	def __init__(self, clients, config):
		self.clients = clients
		self.config = config

		# Data
		resize = A.Compose(
			[
				A.Resize(height=self.config['image_height'], width=self.config['image_width']),
			],
		)

		self.dataset_valid = PlantDataset(set_dir='valid', client_dir=None, transform=resize)
		self.loader_valid = torch.utils.data.DataLoader(dataset=self.dataset_valid, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True, num_workers=2)

		# Model
		self.model = CNN().to(device)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

		# Criterion
		self.criterion = nn.BCELoss()

	def send_to_clients(self):
		for client in self.clients:
			client.receive_from_server(self.model.state_dict(), self.optimizer.state_dict())

	def receive_from_clients(self, k):
		# 1: Aggregate the weights
		model_state_dict_total, optimizer_state_dict_total = clients[0].send_to_server()

		for client in clients[1:]:
			model_state_dict, optimizer_state_dict = client.send_to_server() # Receive weights of the local models from clients

			for layer in model_state_dict:
				model_state_dict_total[layer] = model_state_dict_total[layer].float() + model_state_dict[layer]

		# 2: Update the global model
		for layer in model_state_dict_total:
			model_state_dict_total[layer] = model_state_dict_total[layer].float() / len(clients)

		self.model.to(device)

		# 3: Validation loop
		self.model.load_state_dict(model_state_dict_total, strict=True)
		self.optimizer.load_state_dict(optimizer_state_dict_total)

		valid_loss = 0
		total, correct = 0, 0
		self.model.eval()
		with torch.no_grad():
			for inputs_color, inputs_side, labels in tqdm(self.loader_valid, position=1, desc='Batch valid', leave=False, dynamic_ncols=True):
				
				if self.config['image_type'] == 'color':
					inputs = inputs_color
				elif self.config['image_type'] == 'side':
					inputs = inputs_side
				inputs = inputs.float().to(device)
				labels = labels.float().to(device)

				# Forward pass ➡
				preds = self.model(inputs).squeeze()
				loss = self.criterion(preds, labels)
				valid_loss += loss

				# Accuracy
				total += labels.size(0)
				correct += (torch.round(preds) == torch.round(labels)).sum().item()

		self.model.to('cpu')

		# Save model
		checkpoint = {
			"state_dict": self.model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
		}
		filename = f'./runs/test.pth.tar'
		save_checkpoint(checkpoint, filename)

		# Log losses and scores
		f = open('sample.txt', 'a')
		f.write(f'round: {k},\t valid_loss: {valid_loss},\t accuracy: {correct/total},\n')
		f.close()

if __name__ == '__main__':
	config = dict(
		epochs=1,
		batch_size=8,
		criterion='BinaryCrossEntropy', # CrossEntropy, BinaryCrossEntropy
		optimizer='SGD', # SGD
		learning_rate=0.001,
		momentum=0.9,
		dataset='Plant',
		image_type='color', # color, side
		data_augmentation=True,
		image_height=480, # 480, 960
		image_width=640, # 640, 1280
	)

	client_directories = ['A1', 'A2', 'A4', 'A6', 'B1', 'B2', 'B3', 'B4']
	clients = []
	for client_dir in client_directories:
		clients.append(Client(client_dir, config))


	# Server
	server = Server(clients, config)

	def hfl(rounds):
		for k in tqdm(range(rounds), position=0, desc='Round', leave=False, dynamic_ncols=True):
			server.send_to_clients()

			for client in tqdm(clients, position=1, desc='Client', leave=False, dynamic_ncols=True):
				client.train()
			
			server.receive_from_clients(k)

	hfl(rounds=10)