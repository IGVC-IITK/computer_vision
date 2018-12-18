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
from threshold import thresholdModel
# from advanced_lane_detection.advanced import advancedModel
from fit import drawLane
c=0
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
    t2 = thresholdModel(cv_image)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(t2,"8UC1"))
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
