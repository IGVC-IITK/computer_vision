

# from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.fftpack as fp
import numpy as np
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import pickle
from time import time
import json
import time
from PIL import Image

nwindows=20

polydeg=3

def denoise(mask,kernel_size,iterations):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    for i in range(iterations):
        mask = cv2.erode(mask, element, iterations = 1)
        mask = cv2.erode(mask, element, iterations = 1)
        mask = cv2.dilate(mask, element, iterations = 1)
        mask = cv2.dilate(mask, element, iterations = 1)
    return mask


def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:])+midpoint

	# Choose the number of sliding windows
	# Defined as global

	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 100
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, polydeg)
	right_fit = np.polyfit(righty, rightx, polydeg)
	# print("DONE polyfit")
	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['left_x']=leftx
	ret['left_y']=lefty
	ret['right_x']=rightx
	ret['right_y']=righty
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def drawLane(img):
	mask = np.zeros(shape=(img.shape[0],img.shape[1]))
	mask [0:120,:]=1
	mask = mask == 0
	i=img*mask
	i=denoise(i,2,1)
	res = line_fit(i)

	f=Image.fromarray(i)
	# f.show()


	left_w=np.zeros(shape=(1,polydeg+1))
	right_w=np.zeros(shape=(1,polydeg+1))
	left_w[0] = res['left_fit'] # order is x^polydeg ,... , x^2 , X^1 , x^0
	right_w[0] = res['right_fit']

	left_lane=[]
	right_lane=[]
	num=nwindows
	left_x =  np.arange(0, img.shape[0], int(img.shape[0]/num))
	right_x =  np.arange(0, img.shape[0], int(img.shape[0]/num))

	left_features = np.zeros(shape=(num,polydeg+1))
	for j in range(polydeg):
		left_features[:,j] = left_x**(polydeg-j)
	left_features[:,polydeg] = np.ones(num)

	right_features = np.zeros(shape=(num,polydeg+1))
	right_features[:,polydeg] = np.ones(num)
	for j in range(polydeg):
		right_features[:,j] = right_x**(polydeg-j)
	right_features[:,polydeg] = np.ones(num)

	y_left = np.dot(left_w,np.transpose(left_features))
	y_right = np.dot(right_w,np.transpose(right_features))

	pts_left = np.zeros(shape=(2,num))
	pts_left[0] = y_left
	pts_left[1] = left_x
	pts_left=np.transpose(pts_left)

	pts_right = np.zeros(shape=(2,num))
	pts_right[0] = y_right
	pts_right[1] = right_x
	pts_right=np.transpose(pts_right)

	pts_left = pts_left.astype(int)
	pts_right = pts_right.astype(int)

	# for i in range(pts_left.shape[0]):
	# 	print(pts_left, "   ",pts_right)

	pts_left = pts_left.reshape((-1,1,2))
	pts_right = pts_right.reshape((-1,1,2))

	leftx=res['left_x']
	lefty=res['left_y']
	rightx=res['right_x']
	righty=res['right_y']

	# print(leftx.shape[0],"     ", lefty.shape[0])
	# print(rightx.shape[0],"     ", righty.shape[0])


	left = np.zeros(shape=(2,leftx.shape[0]))
	left[0] = leftx
	left[1] = lefty
	left=np.transpose(left)

	right = np.zeros(shape=(2,rightx.shape[0]))
	right[0] = rightx
	right[1] = righty
	right=np.transpose(right)

	left = left.astype(int)
	right = right.astype(int)


	left = left.reshape((-1,1,2))
	right = right.reshape((-1,1,2))

	llane = np.zeros(shape=(img.shape[0],img.shape[1]))
	rlane = np.zeros(shape=(img.shape[0],img.shape[1]))

	right_max_pt=max(righty)
	right_min_pt=min(righty)
	left_max_pt=max(lefty)
	left_min_pt=min(lefty)

	#using polyfit
	llane=cv2.polylines(llane,[pts_left],False,(255,255,255),2)
	llane[:left_min_pt,:]=0
	llane[left_max_pt:,:]=0
	rlane=cv2.polylines(rlane,[pts_right],False,(255,255,255),2)
	rlane[:right_min_pt,:]=0
	rlane[right_max_pt:,:]=0
        rlane[right_max_pt:,:]=0
	gl=Image.fromarray(llane)
	gr=Image.fromarray(rlane)
	t=res['out_img']
	h=Image.fromarray(t)
	img = cv2.addWeighted(llane,1,rlane,1,0)
	return img,t


#img=cv2.imread('gs.jpg',0)
#img = cv2.resize(img,(640,480))
#c,d=drawLane(img)
#plt.imshow(d)
#plt.show()
