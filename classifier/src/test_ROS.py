import numpy as np
from sklearn.externals import joblib
import rospy
from classifier.srv import *

clf= joblib.load('lane2.pkl')

def handle_classify(req):
	d=np.zeros(27*576)
	for i in range(27*576):
	    d[i]=req.data[i]
        print len(d)
        X=np.zeros(shape=(576,27))
        for k in range (576):
            X[k]=d[27*k:27*k+27]
        print X
	y=clf.predict(X)
        print y
	return lane_classifierResponse(y)

def classifier_server():
	rospy.init_node('classifier_server')
	s = rospy.Service('classifier',lane_classifier, handle_classify)
	print "server ready "
	rospy.spin()

if __name__ == "__main__":
	classifier_server()
