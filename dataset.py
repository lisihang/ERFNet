import os
import cv2
import PIL
import torch
import numpy as np
from torchvision import transforms
from torch.utils import data
import copy

class Cityscapes(data.Dataset):
	def __init__(self, root, subset, transform):
		self.images_root = os.path.join(root, 'leftImg8bit/')
		self.labels_root = os.path.join(root, 'gtFine/')
		
		self.images_root += subset
		self.labels_root += subset

		self.images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if self.is_image(f)]
		self.images.sort()

		self.labels = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if self.is_label(f)]
		self.labels.sort()

		self.transform = transform

	def __getitem__(self, index):
		image = self.load(self.images[index]).convert('RGB')
		label = self.load(self.labels[index]).convert('L')
		image, label = self.transform(image, label)
		
		return image, label

	def __len__(self):
		return len(self.images)

	def load(self, path): 
		return PIL.Image.open(path)

	def is_image(self, filename):
		return filename.endswith('.png')

	def is_label(self, filename):
		return filename.endswith('labelTrainIds.png')
