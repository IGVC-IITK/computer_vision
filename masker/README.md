# Masker Package for Vision Pipeline
This package contains the masker_node that subscribes to labels of each pixel from gslicr_ros_node and predictions for each superpixel from classifier_node and gives a binary image where white patches depicts the parts which are lanes and black parts describes the parts of the image which are non lane.

# Parameters
It takes input following parameters from ROS params :-
### 1.found_total_superpixels:
This , as the name suggests is the total number of superpixels in the image.
### 2. Value of N:
It is the size of the desired matrix of superpixels from gSLICr, however the found_total_superpixels may noy always equal to n^2 ( This is an ambiguity in the algorithm itself )
