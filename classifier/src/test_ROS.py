import numpy as np
from sklearn.externals import joblib
import rospy
from classifier.srv import *

clf= joblib.load('lane2.pkl')
inner_superpixels=576
def handle_classify(req):
	d=np.zeros(27*inner_superpixels)
	for i in range(27*inner_superpixels):
	    d[i]=req.data[i]
        print len(d)
        X=np.zeros(shape=(inner_superpixels,27))
        for k in range (inner_superpixels):
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
