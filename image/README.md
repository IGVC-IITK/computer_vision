# Classical_Vision_Pipeline
An alternate classical vision based pipeline to obtain a binary image showing lanes and not-lane part of the image.
This pipeline uses standard algorithms like blurring,thresholding,denoising and selecting ROI to do the needful task.
## Architecture
  ### handle_classify.py && threshold.py
  The node `handle_classify.py` subscribes to an unwarped RGB image from the topic `\cv_camera\image_raw` and publishes an       unwarped predicted image to topic `\binary_unwarped`. It first converts subscribed image to hsv and then thresholds it         followed by removing the noise by applying `erode` filter. 
  ### Output obtained after thresholding
  The following images show the output of the image for the given RGB image.
  ### fit.py
` fit.py` polyfits the lane in top view image so that lane obtained remains continuous. 
  ### Output obtained after polyfitting
  The following images show the output of the image for the given RGB image.
