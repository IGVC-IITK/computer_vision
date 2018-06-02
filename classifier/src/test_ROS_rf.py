#!/usr/bin/env python
from sklearn.externals import joblib
import pickle
import rospy
import os
from classifier.srv import *
#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
random_forest = joblib.load(os.path.join(__location__, 'rforest_big_full_USA.pkl'))
features=75
superpixels=32*18

def handle_classify(req):
	d=np.zeros(features*superpixels)
	for i in range(features*superpixels):
		d[i]=req.data[i]
	#print len(d)
	X=np.zeros((superpixels,features))
	for k in range(superpixels):
		X[k]=d[features*k:features*k+features]
	#print X
	y=random_forest.predict(X)
	#print y
	return lane_classifierResponse(y)

def classifier_server():
	rospy.init_node('classifier_server')
	s = rospy.Service('classifier',lane_classifier, handle_classify)
	print "server ready "
	rospy.spin()

if __name__ == "__main__":
	classifier_server()
