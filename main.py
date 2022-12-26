
import cv2
import numpy as np

from model import *

cap=cv2.VideoCapture(1)
dataset=data_read_and_processing('data.csv')
data_x,data_y=data_train_model(dataset)
model=neighbors.KNeighborsClassifier(n_neighbors=5,p=2)
model.fit(data_x,data_y)

sum=[0,0,0]
# i=0

while True:
	_,Frame=cap.read()
	h,s,v=cv2.split(cv2.cvtColor(Frame,cv2.COLOR_BGR2HSV))

	_, thresh1 = cv2.threshold(h, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# element in openning
	kernel = np.ones((5, 5), np.uint8)
	# opening closing
	opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)

	try:
		sum = np.array([rgb_image(Frame, closing)])
		label = model.predict(sum)
		if label==0:
			print()
		else:
			# print(sum)
			# label=predict_classification(sum, dataset, 5)
			# print(label)
			if label==1:
				print('red')
			elif label==2:
				print('blue')
			elif label==3:
				print('green')

	except:
		pass

	cv2.imshow('hue',h)
	cv2.imshow('hsv',cv2.cvtColor(Frame,cv2.COLOR_BGR2HSV))
	cv2.imshow('locnhieu',closing)
	cv2.imshow('web_cam',Frame)



	key = cv2.waitKey(1)
	if key == 27:
		break