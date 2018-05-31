from dataset import Cityscapes
from erfnet import ERFNet
from transform import *

import PIL
import cv2
import fire
import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchnet import meter
from torchvision.transforms import ToPILImage

batch_size = 2
num_classes = 20
num_epochs = 20
weight = torch.ones(num_classes)
weight[0] = 2.8149201869965
weight[1] = 6.9850029945374
weight[2] = 3.7890393733978
weight[3] = 9.9428062438965
weight[4] = 9.7702074050903
weight[5] = 9.5110931396484
weight[6] = 10.311357498169
weight[7] = 10.026463508606
weight[8] = 4.6323022842407
weight[9] = 9.5608062744141
weight[10] = 7.8698215484619
weight[11] = 9.5168733596802
weight[12] = 10.373730659485
weight[13] = 6.6616044044495
weight[14] = 10.260489463806
weight[15] = 10.287888526917
weight[16] = 10.289801597595
weight[17] = 10.405355453491
weight[18] = 10.138095855713
weight[19] = 0
weight = weight.cuda()

def IoU(output, labels):
	output = output.view(-1)
	labels = labels.view(-1)
	iou = (labels == output).float().sum() / (0.0 + labels.size(0))
	return iou

def draw(index, output):
	img = Colorize()(output.unsqueeze(0))
	img = ToPILImage()(img)
	img.save('./pic/'+str(index)+'.png')

def load_my_state_dict(model, state_dict):
	own_state = model.state_dict()
	for name, param in state_dict.items():
		print(name)
		if name not in own_state:
			continue
		own_state[name].copy_(param)
	return model

def eval(epoch):
	val_dataset = Cityscapes(root='.', subset='val', transform=MyTransform(augment=False))
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

	erfnet = torch.load('./model/erfnet'+str(epoch)+'.pth')
	erfnet.cuda()
	erfnet.eval()

	iou_meter = meter.AverageValueMeter()

	for i, (images, labels) in enumerate(val_loader):
		images = Variable(images.cuda(), volatile=True).cuda()
		labels = Variable(labels.cuda()).cuda()

		outputs = erfnet(images)
		iou = IoU(outputs.max(1)[1].data, labels.data)
		iou_meter.add(iou)
		draw(i, outputs[0].max(0)[1].byte().cpu().data)

	iou_avg = iou_meter.value()[0]
	print("Val_IoU : {:.5f}".format(iou_avg))

def train():
	train_dataset = Cityscapes(root='.', subset='train', transform=MyTransform())
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

	erfnet = ERFNet(num_classes)

	optimizer = Adam(erfnet.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
	lambda1 = lambda epoch: pow((1-((epoch-1)/num_epochs)),0.9)
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
	criterion = nn.NLLLoss2d(weight)
	iou_meter = meter.AverageValueMeter()

	erfnet.cuda()
	erfnet.train()

	for epoch in range(num_epochs):
		print("Epoch {} : ".format(epoch + 1))
		total_loss = []
		iou_meter.reset()
		scheduler.step(epoch)
		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images.cuda(), requires_grad=True).cuda()
			labels = Variable(labels.cuda()).cuda()

			outputs = erfnet(images)

			optimizer.zero_grad()
			loss = criterion(torch.nn.functional.log_softmax(outputs), labels)
			loss.backward()
			optimizer.step()

			iou = IoU(outputs.max(1)[1].data, labels.data)
			iou_meter.add(iou)
			total_loss.append(loss.data[0] * batch_size)

		iou_avg = iou_meter.value()[0]
		loss_avg = sum(total_loss) / len(total_loss)
		scheduler.step(loss_avg)
		print("IoU : {:.5f}".format(iou_avg))
		print("Loss : {:.5f}".format(loss_avg))
		torch.save(erfnet, './model/erfnet'+str(epoch)+'.pth')

		eval(epoch)

def work():
	erfnet = torch.load('./model/erfnet7.pth')
	erfnet.cuda()
	erfnet.eval()

	videoCapture  = cv2.VideoCapture("test.mp4")
	fps = videoCapture.get(cv2.CAP_PROP_FPS)
	videoWriter = cv2.VideoWriter('t.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (1024, 512))
	success, frame = videoCapture.read()
	while success:
		frame = cv2.resize(frame, (1024, 512))
		image = ToTensor()(frame)
		image = Variable(image.cuda(), volatile=True).unsqueeze(0).cuda()
		output = erfnet(image)
		img = Colorize()(output[0].max(0)[1].byte().cpu().data.unsqueeze(0))
		img = ToPILImage()(img)
		img = np.array(img)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		videoWriter.write(img)
		success, frame = videoCapture.read()

if __name__ == '__main__':
	fire.Fire()