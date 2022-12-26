from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.metrics import accuracy_score # for evaluating results
import csv
import cv2

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return dataset

def string_data_to_float(dataset):
	for i in range(len(dataset[0]) ):
		str_column_to_float(dataset, i)

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def rgb_image(image,image_binary):
    h, w = image_binary.shape
    sum=[0,0,0]
    count=0
    for i in range(h):
        for j in range(w):
            if image_binary[i][j] == 255:
                sum[0] += image[i][j][0]
                sum[1] += image[i][j][1]
                sum[2] += image[i][j][2]
                count += 1

    sum[0] = int ( sum[0] / count )
    sum[1] = int ( sum[1] / count )
    sum[2] = int ( sum[2] / count )
    # print(sum)
    # print(time.time() - t1)
    return sum

def data_read_and_processing(file):
    dataset=load_csv(file)

    string_data_to_float(dataset)
    return dataset

def data_train_model(dataset):
    data_x = np.array([row[0:-1] for row in dataset])
    data_y = np.array([row[-1] for row in dataset])
    return data_x,data_y

dataset=data_read_and_processing('data.csv')

data_x,data_y=data_train_model(dataset)



if __name__=="__main__":

    model=neighbors.KNeighborsClassifier(n_neighbors=5,p=2)
    model.fit(data_x,data_y)

    image_test=cv2.imread('test1.png',1)
    image_test_hsv=cv2.cvtColor(image_test,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(image_test_hsv)

    _,thresold1=cv2.threshold(h,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    openning=cv2.morphologyEx(thresold1,cv2.MORPH_OPEN,kernel)
    closing=cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel)

    x_test=np.array([rgb_image(image_test,closing)])
    y_pred=model.predict(x_test)
    print(y_pred)

