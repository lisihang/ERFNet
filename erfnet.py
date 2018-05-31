import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, out_channels - in_channels, (3, 3), stride=2, padding=1, bias=True)
		self.pool = nn.MaxPool2d(2, stride=2)
		self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

	def forward(self, x):
		out = torch.cat([self.conv(x), self.pool(x)], 1)
		out = self.bn(out)
		return F.relu(out)

class Non_bottleneck_1D(nn.Module):
	def __init__(self, channels, dropprob, dilated):
		super().__init__()

		self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1, 0), bias=True)
		self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1), bias=True)

		self.bn1 = nn.BatchNorm2d(channels, eps=1e-3)

		self.conv3x1_2 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(dilated, 0), bias=True, dilation=(dilated, 1))
		self.conv1x3_2 = nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, dilated), bias=True, dilation=(1, dilated))

		self.bn2 = nn.BatchNorm2d(channels, eps=1e-3)

		self.dropout = nn.Dropout2d(dropprob)

	def forward(self, x):
		out = self.conv3x1_1(x)
		out = F.relu(out)
		out = self.conv1x3_1(out)
		out = self.bn1(out)
		out = F.relu(out)

		out = self.conv3x1_2(out)
		out = F.relu(out)
		out = self.conv1x3_2(out)
		out = self.bn2(out)

		if self.dropout.p != 0:
			out = self.dropout(out)

		return F.relu(out + x)

class Encoder(nn.Module):
	def __init__(self):
		super().__init__()

		self.initial_block = DownsamplerBlock(3, 16)

		self.layers = nn.ModuleList()

		self.layers.append(DownsamplerBlock(16, 64))

		for i in range(5):
			self.layers.append(Non_bottleneck_1D(64, 0.03, 1))

		self.layers.append(DownsamplerBlock(64, 128))

		for i in range(2):
			self.layers.append(Non_bottleneck_1D(128, 0.3, 2))
			self.layers.append(Non_bottleneck_1D(128, 0.3, 4))
			self.layers.append(Non_bottleneck_1D(128, 0.3, 8))
			self.layers.append(Non_bottleneck_1D(128, 0.3, 16))

	def forward(self, x):
		out = self.initial_block(x)

		for layer in self.layers:
			out = layer(out)

		return out

class UpsamplerBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=True)
		self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		return F.relu(out)

class Decoder(nn.Module):
	def __init__(self, classes):
		super().__init__()

		self.layers = nn.ModuleList()

		self.layers.append(UpsamplerBlock(128, 64))
		self.layers.append(Non_bottleneck_1D(64, 0, 1))
		self.layers.append(Non_bottleneck_1D(64, 0, 1))

		self.layers.append(UpsamplerBlock(64, 16))
		self.layers.append(Non_bottleneck_1D(16, 0, 1))
		self.layers.append(Non_bottleneck_1D(16, 0, 1))

		self.output_conv = nn.ConvTranspose2d(16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)

	def forward(self, x):
		out = x

		for layer in self.layers:
			out = layer(out)

		out = self.output_conv(out)

		return out

class ERFNet(nn.Module):
	def __init__(self, classes, encoder=None):
		super().__init__()

		if encoder != None:
			self.encoder = encoder
		else:
			self.encoder = Encoder()
		self.decoder = Decoder(classes)

	def forward(self, x):
		out = self.encoder(x)
		out = self.decoder(out)

		return out
