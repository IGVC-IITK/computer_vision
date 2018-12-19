import numpy as np
import cv2
from PIL import Image

# Choose the number of sliding windows
nwindows=20
# Choose the degree of polynomial line fitting
polydeg=3

def line_fit(binary_warped):
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
	ret['left_max_pt']=max(lefty)
	ret['left_min_pt']=min(lefty)
	ret['right_max_pt']=max(righty)
	ret['right_min_pt']=min(righty)
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def drawLane(img):
	res = line_fit(img)
	# Polyfit equation constants
	left_w=np.zeros(shape=(1,polydeg+1))
	right_w=np.zeros(shape=(1,polydeg+1))
	left_w[0] = res['left_fit'] 
	right_w[0] = res['right_fit'] # Order is x^polydeg ,... , x^2 , X^1 , x^0
	num=nwindows
	# X coordinate of final lane points
	left_x =  np.arange(0, img.shape[0], int(img.shape[0]/num))
	right_x =  np.arange(0, img.shape[0], int(img.shape[0]/num))
	# Making blank polynomial matrix
	left_features = np.zeros(shape=(num,polydeg+1))
	right_features = np.zeros(shape=(num,polydeg+1))
	for j in range(polydeg): 
		left_features[:,j] = left_x**(polydeg-j)
		right_features[:,j] = right_x**(polydeg-j)
	left_features[:,polydeg] = np.ones(num)
	right_features[:,polydeg] = np.ones(num)
	# Y coordinate of final lane points
	left_y = np.dot(left_w,np.transpose(left_features)) 
	right_y = np.dot(right_w,np.transpose(right_features))
	# List of all left lane points
	pts_left = np.zeros(shape=(2,num))
	pts_left[0] = left_y
	pts_left[1] = left_x
	pts_left=np.transpose(pts_left)
	# List of all right lane points
	pts_right = np.zeros(shape=(2,num))
	pts_right[0] = right_y
	pts_right[1] = right_x
	pts_right=np.transpose(pts_right)
	# Necessary conversions
	pts_left = pts_left.astype(int)
	pts_right = pts_right.astype(int)
	pts_left = pts_left.reshape((-1,1,2))
	pts_right = pts_right.reshape((-1,1,2))
	# Declare blank lanes
	llane = np.zeros(shape=(img.shape[0],img.shape[1]))
	rlane = np.zeros(shape=(img.shape[0],img.shape[1]))
	# Using polyfit
	llane=cv2.polylines(llane,[pts_left],False,(255,255,255),2) # Left lane
	rlane=cv2.polylines(rlane,[pts_right],False,(255,255,255),2) # Right lane
	# Upper and lower end points of lanes
	ymax=img.shape[1]
	right_max_pt=res['right_max_pt']
	right_min_pt=res['right_min_pt']
	left_max_pt=res['left_max_pt']
	left_min_pt=res['left_min_pt']
	# Removing upper and lower extrapolations
	llane[:max(0,left_min_pt-1),:]=0 
	llane[min(left_max_pt+1,ymax):,:]=0
	rlane[:max(0,right_min_pt-1),:]=0
	rlane[min(right_max_pt+1,ymax):,:]=0
	# Display Normal Image for testing
	f=Image.fromarray(img)
	# f.show()
	# Display Rectangle Image for testing
	t=res['out_img']
	h=Image.fromarray(t)
	# h.show()
	# Adding left and right lanes to one image
	img = cv2.addWeighted(llane,1,rlane,1,0)
	return img,t

########################################## For testing only ###################################

# img=cv2.imread('top_view_out.jpg',0)
# img_lane=drawLane(img)
# img_lane=img_lane.astype(np.uint8)
# img_orig=cv2.imread('top_view.jpg',0)
# img_orig=cv2.resize(img_orig,(img_orig.shape[1]//2,img_orig.shape[0]//2))
# img_orig=img_orig.astype(np.uint8)
# img_comb=cv2.addWeighted(img_orig,0.5,img_lane,0.5,0)
# disp_comb=Image.fromarray(img_comb)
# disp_comb.show()
