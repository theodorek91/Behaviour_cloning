import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

Keys=['center_image','left_image','right_image','center_steering','left_steering','right_steering','throttle','brake','speed']
Data=dict.fromkeys(Keys)
for keys in Data:
	Data[keys]=[]
Data['center_image']=[]
with open("ot\driving_log.csv",'rt') as f:
		reader = csv.reader(f)
		for line in reader:
			Data['center_image'].append(plt.imread(np.asarray(line)[0]))
			Data['left_image'].append(plt.imread(np.asarray(line)[1]))
			Data['right_image'].append(plt.imread(np.asarray(line)[2]))
			Data['center_steering'].append(float(np.asarray(line)[3]))
			Data['left_steering'].append(float(np.asarray(line)[3])-0.2)
			Data['right_steering'].append(float(np.asarray(line)[3])+0.2)
			Data['throttle'].append(float(np.asarray(line)[4]))
			Data['brake'].append(float(np.asarray(line)[5]))
			Data['speed'].append(float(np.asarray(line)[6]))
X_train = sum([Data['center_image'],Data['left_image'],Data['right_image']],[])


List1 =[]
List1.extend((Data['center_steering'],Data['throttle'],Data['brake'],Data['speed']))
List1=np.transpose(List1)
List2 =[]
List2.extend((Data['left_steering'],Data['throttle'],Data['brake'],Data['speed']))
List2=np.transpose(List2)
List3 =[]
List3.extend((Data['right_steering'],Data['throttle'],Data['brake'],Data['speed']))
List3=np.transpose(List3)

Y_train=[]
Y_train =(np.concatenate((List1,List2,List3)))
print (np.asarray(Y_train).shape)
print((np.asarray(X_train).shape))

