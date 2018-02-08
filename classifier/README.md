# Classifier Package for Vision Pipeline
This package contains : <br/>
	1. classifier_node <br/>
	2. launchfile for vision pipeline <br/>
	3. trained neural network parameters

# Description
### classifier_node :
Takes ros parameter **found_total_superpixels** which is the no. of superpixels in which the image was segmented. 

This node subscribes to the topic "/gslicr/averages" to get the segmented and averaged image (does not use CUDA or gslicr) of size **N\*N**(may or may not be equal to found_total_superpixels which is an ambiguity in the gslicr algorithm to be verified) .

Creates a window of 5 x 5 for every superpixel for storing the rgb values of a superpixel and all its neighbours within two layers of it. *Green* padding is done for boundary superpixels. Hence, 75 features are generated for each superpixel in the image for the neural network.

Uses **ANN_MLP (Multi Layer Perceptron)** machine learning algorithm in OpenCV Library. Pre-trained parameters are loaded from *mlp.yml* present in /src . 

Prediction array is then published on the ros topic **/predictions** which is finally used by *masker* to give final binary image of lane.
