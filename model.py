#import required packages
import numpy as np
import keras
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping as ES #ES(monitor='val_loss', min_delta=0.11, patience=5, verbose=0, mode='max')
from keras.preprocessing.image import ImageDataGenerator
import csv
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model
import matplotlib.pyplot as plt

#since AWS instance does not have UI for plot
plt.switch_backend('agg')

#Required variable Definitions
batchsize = 50
nepoch = 50

#Function definitions

def image_func(img):  
	#pre processing. _ Did not work. Worked for training but on simulator complained about shape of input. 
	#Was meant to add extra layer to input image of Canny transform so that gradiest are taken into consideration during training to provide better accuracy.
	img2=cv2.Canny(img,threshold1=200,threshold2=200) 
	img3=np.concatenate((img,np.reshape(img2,(160,320,1))),axis=2)
	print(np.asarray(img3).shape)
	yield img3

#load database
def load_data():
	[X_train,Y_train]=pickle.load(open('/home/carnd/CarND/data3.p','rb'))

	#to load data from the training images since large, use pickle file.
	'''
	lines=[]
	with open('/home/carnd/CarND/ot/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	images=[]
	measurements = []
	for line in lines:
		for i in range(3):
			source_path=line[i]
			filename=source_path.split('\\')[-1]
			current_path='/home/carnd/CarND/ot/IMG/'+ filename
			image=cv2.imread(current_path)
			images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
			if i ==0:
				measurement = float(line[3])
				measurements.append(measurement)
			if i ==1:
				measurement = float(line[3])+0.15
				measurements.append(measurement)
			if i==2:
				measurement = float(line[3]) -0.15
				measurements.append(measurement)
	X_train =np.array(images)
	Y_train = np.array(measurements)
	'''
	print(X_train.shape,Y_train.shape)
	return X_train,Y_train
	

	
if __name__ == '__main__':
	print("loading data")
	X_train,Y_train=load_data()
	
	#print("Augmentation") #pre-processing : Normalize   
	#this part didnt work kept giving memory error due to large dataset.
	#datagen = ImageDataGenerator(samplewise_std_normalization=True)	
	#datagen.fit(np.asarray(X_train))

	print("splitting data")
	(XTrain,XVal,YTrain,YVal)=train_test_split(X_train,Y_train,test_size=0.2,random_state=666)
	
	'''
	Attempt1 - Implementation of modified Alexnet Layer pattern.
	(oringinal Alexnet layer information from : https://github.com/dmlc/minerva/wiki/Walkthrough:-AlexNet)
	5 (Conv) + 1 (dropout) + 3 (Full) layers
	Layer 1 - Convolution2D 
				Max
				Batch Normalization
	Layer 2 - Convolution2D
				Max
				Batch Normalization
	Layer 3 - Convolution2D
	Layer 4 - Convolution2D
	Layer 5 - Convolution2D
				Max

	Layer 6 - Dropout (to prevent overfitting)

	Layer 7 - Full (Dense)	
	Layer 8 - Full (Dense)	
	Layer 9 - Full (Dense)	

	Also, Callbacks implemented to monitor Validation accuracy to stop training if value stops increasing.

	'''
	#for Early stopping of model incase loss stopped decreasin in order of 0.001 0r 0.1 percent
	callbacks = [ES(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='min')]

	
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((50,20), (0,0))))
	model.add(Convolution2D(96,11,11,subsample= (4,4),activation='relu'))
	model.add(ZeroPadding2D())
	model.add(MaxPooling2D(pool_size=(3,3),strides=(4,4)))
	model.add(ZeroPadding2D())
	model.add(BatchNormalization())
	model.add(Convolution2D(256,5,5,subsample= (1,1),activation='relu')) 
	model.add(ZeroPadding2D())
	model.add(MaxPooling2D(pool_size=(3,3),strides=(4,4)))
	model.add(ZeroPadding2D())
	model.add(BatchNormalization())
	model.add(Convolution2D(384,3,3,subsample= (1,1),activation='relu'))
	model.add(ZeroPadding2D())
	model.add(Convolution2D(384,3,3,subsample= (1,1),activation='relu'))
	model.add(ZeroPadding2D())
	model.add(Convolution2D(256,3,3,subsample= (1,1),activation='relu'))
	model.add(ZeroPadding2D())
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(ZeroPadding2D())
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation= 'relu'))
	model.add(Dense(1))
	model.summary()
	
	print("training model")
	model.compile(loss='mse',optimizer='adam')
	history = model.fit(XTrain, YTrain, batch_size=batchsize,epochs=50, verbose=1, callbacks=callbacks, validation_data=(XVal, YVal))

	plot_model(model, to_file='model.png', show_shapes='True')
	print(history.history['loss'])
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper left')
	plt.savefig('lossHistory.png')
	print("Saving Model")
	model.save("model.h5")
	print("model saved")
	print("   ....    ")
	print("End of Script")
	
