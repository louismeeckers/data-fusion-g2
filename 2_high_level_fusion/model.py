import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
	def __init__(self, in_channels=3):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer4 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(15*20*256, 512),
			nn.ReLU())
		self.fc1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(512, 512),
			nn.ReLU())
		self.fc2= nn.Sequential(
			nn.Linear(512, 1))
		
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = x.reshape(x.size(0), -1)
		x = self.fc(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = torch.sigmoid(x)
		return x

if __name__ == '__main__':
	model = CNN(in_channels=3)
	summary(model, input_size=(3, 480, 640), batch_size=8, device='cpu')