import os
import re
from PIL import Image
import numpy as np
'''color = [(0,0,0),(70,130,180),(70,70,70),(107,142,35),(152,251,152),(128,64,128),(153,153,153),(220,220,0),(0,0,142),(244,35,232),(119,11,32),(220,20,60),(255,0,0),(111,74,0),(102,102,156),(250,170,30),(190,153,153),(81,0,81),(0,0,230),(250,170,160),(0,60,100),(150,100,100),(0,0,110),(150,120,90),(0,0,90),(0,0,70),(180,165,180),(230,150,140),(0,80,100)]
s = 29'''
color = []
img = []
palette = [0] * 34
for root, dirs, files in os.walk('./gtFine/train'):
	for file in files:
		if re.match(r'(.*)labelIds.png', file):
			path = os.path.join(root, file)
			image = Image.open(path).convert('L')
			w, h = image.size
			for i in range(w):
				for j in range(h):
					pixel = image.getpixel((i, j))
					if pixel not in color:
						color.append(pixel)
						palette[pixel] = img.getpixel((i, j))
						print(palette)
						print(pixel)
						print(len(color))

		if re.match(r'(.*)color.png', file):
			path = os.path.join(root, file)
			img = Image.open(path).convert('RGB')