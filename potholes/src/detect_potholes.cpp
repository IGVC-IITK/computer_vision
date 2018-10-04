/**
 * @file HoughCircle_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


 #include "ros/ros.h"
 #include <cstdlib>
 #include <iostream>
 #include <fstream>
 #include <time.h> 
 #include "std_msgs/UInt16MultiArray.h"
 
 #include <vector>

 #include <opencv2/calib3d/calib3d.hpp>
 #include <opencv2/core/core.hpp>
 #include <opencv2/highgui/highgui.hpp>
 #include <opencv2/imgproc/imgproc.hpp>

 #include <cv_bridge/cv_bridge.h>
 #include <image_transport/image_transport.h>
 #include <sensor_msgs/Image.h>
 #include <sensor_msgs/image_encodings.h>

using namespace std;
using namespace cv;

namespace
{
    // initial and max values of the parameters of interests.
    const int cannyThresholdInitialValue = 90; // tune them 
    const int accumulatorThresholdInitialValue = 27; // tune them
    const int maxAccumulatorThreshold = 200;
    const int maxCannyThreshold = 255;

    cv::Mat HoughDetection(const Mat& src_gray, const Mat& src_display ,const Mat& src_binary ,int cannyThreshold, int accumulatorThreshold)
    {
        // will hold the results of the detection
        std::vector<Vec3f> circles;
        // runs the actual detection
        HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows/8, cannyThreshold, accumulatorThreshold, 0, 0 );

        // clone the colour, input image for displaying purposes
        Mat display = src_display.clone();
        Mat binary = src_binary.clone();
        
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle( display, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( display, center, radius, Scalar(255,255,255), 3, 8, 0 );
            circle( binary, center, radius, Scalar(255,255,255), 3, 8, 0 );
               
        }

	// imshow("here",display); // see this for cross checking
        // shows the results
        return binary;
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr& imgMessage, cv::Mat& image)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(imgMessage, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    image=cv_ptr->image;
  }

sensor_msgs::ImagePtr imageToROSmsg(cv::Mat img, const std::string encodingType, std::string frameId, ros::Time t)
{
    // this part of the source code has been imported from https://github.com/stereolabs/zed-ros-wrapper
    sensor_msgs::ImagePtr ptr = boost::make_shared<sensor_msgs::Image>();
    sensor_msgs::Image& imgMessage = *ptr;
    imgMessage.header.stamp = t;
    imgMessage.header.frame_id = frameId;
    imgMessage.height = img.rows;
    imgMessage.width = img.cols;
    imgMessage.encoding = encodingType;
    int num = 1; //for endianness detection
    imgMessage.is_bigendian = !(*(char *) &num == 1);
    imgMessage.step = img.cols * img.elemSize();
    size_t size = imgMessage.step * img.rows;
    imgMessage.data.resize(size);

    if (img.isContinuous())
        memcpy((char*) (&imgMessage.data[0]), img.data, size);
    else {
        uchar* opencvData = img.data;
        uchar* rosData = (uchar*) (&imgMessage.data[0]);
        for (unsigned int i = 0; i < img.rows; i++) {
            memcpy(rosData, opencvData, imgMessage.step);
            rosData += imgMessage.step;
            opencvData += img.step;
        }
    }
    return ptr;
}



int main(int argc, char** argv)
{
    Mat src, src_gray ;
    cv::Mat binary = cv::Mat::zeros(1170, 1560, CV_8UC3); // 1170 is height and 1560 is width
    ros::init(argc, argv, "potholes");
    ros::NodeHandle n;

    cv::Mat image;
    image.create(cv::Size(1170, 1560), CV_8UC3);
    image_transport::ImageTransport it_avg(n);
    image_transport::Subscriber sub_img = it_avg.subscribe("/top_view", 1, boost::bind(imageCallback, _1, boost::ref(image)));
    
    image_transport::ImageTransport it_pot(n);
    ros::Time t;
    image_transport::Publisher pub_avg = it_pot.advertise("/pothole_image_color", 1);
    image_transport::Publisher binary_potholes = it_pot.advertise("/pothole_image_binary", 1);


    ros::Rate loop_rate(10);

    while(ros::ok())
    {

    ros::spinOnce();
    src = image;

    // Convert it to gray
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    // Reduce the noise so we avoid false circle detection
    GaussianBlur( src_gray, src_gray, Size(11, 11), 2, 2 );

    //declare and initialize both parameters that are subjects to change
    int cannyThreshold = cannyThresholdInitialValue;
    int accumulatorThreshold = accumulatorThresholdInitialValue;
    char key = 0;

        cannyThreshold = std::max(cannyThreshold, 1);
        accumulatorThreshold = std::max(accumulatorThreshold, 1);

        //runs the detection, and update the display
        src = HoughDetection(src_gray, src, binary ,cannyThreshold, accumulatorThreshold);
        string frame_id="camera";
        t = ros::Time::now();
        pub_avg.publish(imageToROSmsg(src, sensor_msgs::image_encodings::BGR8, frame_id, t));
        key = (char)waitKey(10);
    }
    return 0;
}
