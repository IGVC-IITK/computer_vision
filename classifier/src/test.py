import numpy as np
from sklearn.externals import joblib
import rospy
from classifier.srv import *
import time
clf= joblib.load('lane.pkl')
c=time.time()

for j in range (72900):
    X=np.zeros(27)
    for i in range(27):
        X[i]=0
    X=X.reshape(1,-1)
    y=clf.predict(X)
k=time.time()-c
print k


