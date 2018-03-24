import numpy as np
from sklearn.externals import joblib
import rospy
from classifier.srv import *
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

tf.keras.backend.clear_session()
model=Sequential()

model.add(Dense(50, input_dim=75, kernel_initializer='uniform', activation='relu'))
model.add(Dense(30,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dense(30,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dense(30,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dense(15,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dense(15,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dense(15,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dense(1,  kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))

model.load_weights('relu.h5')

features=75
superpixels=1600
x=np.zeros((1600,75))
y=model.predict(x)
print y
