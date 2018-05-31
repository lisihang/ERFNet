import numpy as np
import torch
import random
from PIL import Image, ImageOps
from torchvision.transforms import Resize, ToTensor

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()

class Colorize:
    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

class MyTransform(object):
    def __init__(self, augment=True, height=512):
        self.augment = augment
        self.height = height

    def __call__(self, image, label):
        image = Resize(self.height, Image.BILINEAR)(image)
        label = Resize(self.height, Image.NEAREST)(label)

        if(self.augment):
            hflip = random.random()
            if (hflip < 0.5):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            image = ImageOps.expand(image, border=(transX,transY,0,0), fill=0)
            label = ImageOps.expand(label, border=(transX,transY,0,0), fill=255)
            image = image.crop((0, 0, image.size[0] - transX, image.size[1] - transY))
            label = label.crop((0, 0, label.size[0] - transX, label.size[1] - transY))   

        image = ToTensor()(image)
        label = ToLabel()(label)
        label = Relabel(255, 19)(label)

        return image, label
