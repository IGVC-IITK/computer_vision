#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('image')
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os, glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import unet
global model
model=unet('/home/abhishek/catkin/src/deep/src/model.h5')
model._make_predict_function()
class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("/binary_unwarped",Image,queue_size=1000)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)
    # self.image_sub = rospy.Subscriber("/zed/left/image_rect_color",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    cv_image = cv2.resize(cv_image,(128,128))
    x = cv_image.reshape(1,128,128,3)
    global model
    y = model.predict(x)
    y = y.reshape(128,128)
    y = 255*(y*2555555 > 100)
    y = y.astype(np.uint8)
    y = cv2.resize(y,(640,480))
    y = 255*(y>100)
    y = y.astype(np.uint8)
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(y,"8UC1"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_segmentation_node', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)
