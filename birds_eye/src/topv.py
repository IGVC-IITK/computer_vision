#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from numpy import linalg
import cv_bridge
from sensor_msgs.msg import Image


criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def nothing(x):
    pass

def topview(image,t1,t2):
    if len(image.shape)==2:
        height,width = image.shape
        src = np.array([[0,height/2],[width,height/2],[width,height],[0,height]],dtype='float32')
        dest = np.array([[0,0],[width,0],[width-t1,height-t2],[t1,height-t2]],dtype='float32')
        h, status = cv2.findHomography(src, dest)
        imgd = np.zeros((height,width),dtype='uint8')
        imgd = cv2.warpPerspective(image, h, (width,height))
    else:
        height,width,col = image.shape
        src = np.array([[0,height/2],[width,height/2],[width,height],[0,height]],dtype='float32')
        dest = np.array([[0,0],[width,0],[width-t1,height-t2],[t1,height-t2]],dtype='float32')
        h, status = cv2.findHomography(src, dest)
        imgd = np.zeros((height,width,col),dtype='uint8')
        imgd[:,:,0] = cv2.warpPerspective(image[:,:,0], h, (width,height))
        imgd[:,:,1] = cv2.warpPerspective(image[:,:,1], h, (width,height))
        imgd[:,:,2] = cv2.warpPerspective(image[:,:,2], h, (width,height))
    return imgd,h

rospy.init_node('top_view_maker')
vidFile = cv2.VideoCapture(0)
ret, frame = vidFile.read()
cv2.namedWindow('image')
height,width,_ = frame.shape
cv2.createTrackbar('t1','image',0,width,nothing)
cv2.createTrackbar('t2','image',0,height,nothing)
while ret:
    ret, frame = vidFile.read()
    t1 = cv2.getTrackbarPos('t1','image')
    t2 = cv2.getTrackbarPos('t2','image')
    gray = frame[:,:,0]
    # try:
    #     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    # except Exception as e:
    #     pass
    frame,h = topview(frame,t1,t2)
    cv2.line(frame,(width/2,height/2-25),(width/2,height/2+25),(0,255,0),3)
    cv2.line(frame,(width/2-25,height/2),(width/2+25,height/2),(0,255,0),3)
    # if ret == True:
    #     objpoints.append(objp)
    #     corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    #     imgpoints.append(corners2)
    # frame = cv2.drawChessboardCorners(frame, (7,6), corners,ret)
    cv2.imshow('image',frame)
    key = cv2.waitKey(30)
    if key==27 or key==1048603:
        print h
        np.savetxt('/home/daksh/catkin_ws/top_view.txt',linalg.inv(h),delimiter=', ',newline=';\n',header=' ',comments='')
        break
exit()
