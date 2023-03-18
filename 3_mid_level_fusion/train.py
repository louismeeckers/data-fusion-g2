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
from utils import load_checkpoint, save_checkpoint

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '../data/'


config = dict(
	epochs=30,
	batch_size=8,
	criterion='BinaryCrossEntropy', # CrossEntropy, BinaryCrossEntropy
	optimizer='SGD', # SGD
	learning_rate=0.001,
	momentum=0.9,
	dataset='Plant',
	image_type='color_side', # color, side
	data_augmentation=True,
	image_height=480, # 480, 960
	image_width=640, # 640, 1280
)


def model_pipeline(hyperparameters):

	with wandb.init(project='data-fusion-plant', entity='louismeeckers', config=hyperparameters):
		config = wandb.config
		model, loader_train, loader_valid, criterion, optimizer = make(config)
		train(model, loader_train, loader_valid, criterion, optimizer, config)

	return model


def make(config):
	# Data
	transform = A.Compose(
		[
			A.RandomBrightnessContrast(p=0.3),
			A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
			A.RandomResizedCrop(height=config.image_height, width=config.image_width, scale=(0.6, 1.0), ratio=(0.75, 1.333), interpolation=1, p=0.5),
			A.Resize(height=config.image_height, width=config.image_width),
		],
	)

	resize = A.Compose(
		[
			A.Resize(height=config.image_height, width=config.image_width),
		],
	)

	dataset_train = PlantDataset(set_dir='train', transform=(transform if config.data_augmentation else resize))
	loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)

	dataset_valid = PlantDataset(set_dir='valid', transform=resize)
	loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)

	# Model
	model = CNN().to(device)
	if torch.cuda.device_count() > 1:
		print(f'Train the model on {torch.cuda.device_count()} GPUs')
		model = nn.DataParallel(model)
	model = model.to(device)

	# Criterion and Optimizer
	criterion = build_criterion(config)
	optimizer = build_optimizer(model, config)

	return model, loader_train, loader_valid, criterion, optimizer


def build_criterion(config):
	if config.criterion == 'BinaryCrossEntropy':
		criterion = nn.BCELoss()
	return criterion


def build_optimizer(model, config):
	if config.optimizer == 'SGD':
		optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
	return optimizer


def train(model, loader_train, loader_valid, criterion, optimizer, config):
	# Watch the model: gradients, weights...
	wandb.watch(model, criterion, log='all', log_freq=10)

	example_ct = 0 # number of examples seen
	batch_ct = 0 # number of batches seen

	for epoch in tqdm(range(config.epochs), position=0, desc='Epoch', leave=False, dynamic_ncols=True):
		# Train loop
		train_loss = 0
		model.train()
		for inputs_color, inputs_side, labels in tqdm(loader_train, position=1, desc='Batch train', leave=False, dynamic_ncols=True):
			
			inputs_color = inputs_color.float().to(device)
			inputs_side = inputs_side.float().to(device)
			labels = labels.float().to(device)

			# Forward pass ➡
			preds = model(inputs_color, inputs_side).squeeze()
			loss = criterion(preds, labels)
			
			# Backward pass ⬅
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_loss += loss
			example_ct += len(inputs_color)
			batch_ct += 1

			# Report metrics every 5 batch
			if ((batch_ct + 1) % 5) == 0:
				wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
				tqdm.write(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.8f}")
		
		# Validation loop
		valid_loss = 0
		total, correct = 0, 0
		model.eval()
		with torch.no_grad():
			for inputs_color, inputs_side, labels in tqdm(loader_valid, position=1, desc='Batch valid', leave=False, dynamic_ncols=True):
				
				inputs_color = inputs_color.float().to(device)
				inputs_side = inputs_side.float().to(device)
				labels = labels.float().to(device)

				# Forward pass ➡
				preds = model(inputs_color, inputs_side).squeeze()
				loss = criterion(preds, labels)
				valid_loss += loss

				# Accuracy
				total += labels.size(0)
				correct += (torch.round(preds) == torch.round(labels)).sum().item()

		# Save model
		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		filename = f'./runs/{wandb.run.name}.pth.tar'
		save_checkpoint(checkpoint, filename)

		# Log losses and scores
		wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "accuracy": correct/total}, step=example_ct)


if __name__ == '__main__':
	model = model_pipeline(config)