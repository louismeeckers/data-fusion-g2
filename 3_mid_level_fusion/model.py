import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
	def __init__(self, in_channels=3):
		super(CNN, self).__init__()

		# For Image 1
		self.layer1_1 = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer1_2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer1_3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer1_4 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer1_5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		# For Image 2
		self.layer2_1 = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2_2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2_3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2_4 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2_5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		# For concat
		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(15*20*512, 512),
			nn.ReLU())
		self.fc1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(512, 512),
			nn.ReLU())
		self.fc2= nn.Sequential(
			nn.Linear(512, 1))
		
	def forward(self, x1, x2):
		# Image 1
		x1 = self.layer1_1(x1)
		x1 = self.layer1_2(x1)
		x1 = self.layer1_3(x1)
		x1 = self.layer1_4(x1)
		x1 = self.layer1_5(x1)

		# Image 2
		x2 = self.layer2_1(x2)
		x2 = self.layer2_2(x2)
		x2 = self.layer2_3(x2)
		x2 = self.layer2_4(x2)
		x2 = self.layer2_5(x2)

		x = torch.cat((x1, x2), 1)
		x = x.reshape(x.size(0), -1)
		x = self.fc(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = torch.sigmoid(x)
		return x

if __name__ == '__main__':
	model = CNN(in_channels=3)
	summary(model, input_size=(3, 480, 640), batch_size=8, device='cpu')