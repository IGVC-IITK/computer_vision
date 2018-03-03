import numpy as np
from sklearn.externals import joblib
import rospy
from classifier.srv import *
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(75, input_dim=75, kernel_initializer='uniform', activation='relu'))
model.add(Dense(25,  kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu'))
model.add(Dense(15,  kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu'))
model.add(Dense(1,  kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu'))

model.load_weights('relu_theano.h5')

features=75
superpixels=1600


def handle_classify(req):
	d=np.zeros(features*superpixels)
	for i in range(features*superpixels):
		d[i]=req.data[i]
	#print len(d)
	X=np.zeros((superpixels,features))
	for k in range(superpixels):
		X[k]=d[features*k:features*k+features]
	#print X
	y=model.predict(X)
	#print y
	return lane_classifierResponse(y)

def classifier_server():
	rospy.init_node('classifier_server')
	s = rospy.Service('classifier',lane_classifier, handle_classify)
	print "server ready "
	rospy.spin()

if __name__ == "__main__":
	classifier_server()
