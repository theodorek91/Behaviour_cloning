import numpy as np
import keras
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping as ES #ES(monitor='val_loss', min_delta=0.11, patience=5, verbose=0, mode='max')
from keras.preprocessing.image import ImageDataGenerator

ImageFolderPath = "ot\\IMG\\"
LogPath = "ot\\driving_log.csv"
batchsize = 50
nepoch = 10

#Function definitions

#pre-processing : Resize
def image_process(img):
	return cv2.resize((cv2.cvtColor(img,cv2.Color_RGB2HSV))[:,:,1],(16,32))
	#
	
#load database
def load_data():
	return 0
	

	
if __name__ == '__main__':
	print("loading data")
	print("Augmentation") #pre-processing : Normalize
	datagen = ImageDataGenerator(samplewise_std_normalization=True)	
	datagen.fit(X_train)

	print("splitting data")
	(XTrain,YTrain,XVal,YVal)=train_test_split(X_train,Y_train,test_size=0.2,random_state=666)
	
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
	callbacks = [ES(monitor='val_Accuracy', min_delta=0.11, patience=5, verbose=0, mode='max')]


	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
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
	model.add(Dense(1000, activation='relu'))
	model.add(Activation('softmax'))
	model.summary()
	
	print("training model")
	model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
	history = model.fit_generator(datagen.flow(XTrain, YTrain, batch_size=batch_size), epochs=nepoch, verbose=1, validation_data=(XVal, YVal), callbacks=callbacks)
	
	print("Saving Model")
	model.save_weights("model.h5")
	print("model saved")
	print("   ....    ")
	print("End of Script")
	