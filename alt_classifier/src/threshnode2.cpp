#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <queue>
#include <vector>
#include <iomanip>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/UInt16MultiArray.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

using namespace std;
using namespace cv;
int flag=0;

cv::Mat raw_image;

cv::Mat blurred_image;
cv::Mat final_image;

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

void imageCallback(const sensor_msgs::ImageConstPtr& imgMessage)
  {
    try
    {
      
    raw_image = cv_bridge::toCvShare(imgMessage,"bgr8")->image;
    //cv::imshow("view",raw_image);
    cv::waitKey(30);
      //cv_ptr = cv_bridge::toCvCopy(imgMessage, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }

int main( int argc, char** argv )
{
    ros::init(argc, argv, "thresh_node");
    ros::NodeHandle n;
    image_transport::ImageTransport it_thresh(n);
    ros::Time t;
   
    //image = imread("./frame0003.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
   // image_transport::Subscriber sub_img = it_thresh.subscribe("/top_view", 1, boost::bind(imageCallback, _1, boost::ref(image)));
    image_transport::Subscriber sub_img = it_thresh.subscribe("/top_view",1,imageCallback);
   // ros::spin();

    image_transport::Publisher pub_img = it_thresh.advertise("/final_image", 1);
    ros::Rate loop_rate(1);
    while(ros::ok()){
        ros::spinOnce();
        int width = raw_image.size().width;
        int height = raw_image.size().height;
        if(width!=0 && height!=0){
        cout<<width<<" "<<height<<endl;
        GaussianBlur( raw_image, blurred_image, Size( 15, 15 ), 10, 10 );
     //   cv::imshow("Blurred", blurred_image); 
    

        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                int blue = blurred_image.at<cv::Vec3b>(i,j)[0]; 

                if(blue>155){
                    blurred_image.at<cv::Vec3b>(i,j)[0] = 255;
                    blurred_image.at<cv::Vec3b>(i,j)[1] = 255;
                    blurred_image.at<cv::Vec3b>(i,j)[2] = 255;
                }
                else {
                    blurred_image.at<cv::Vec3b>(i,j)[0] = 0;
                    blurred_image.at<cv::Vec3b>(i,j)[1] = 0;
                    blurred_image.at<cv::Vec3b>(i,j)[2] = 0;
                }
            }
        }
        
        //cv::cvtColor(image, image, CV_RGB2GRAY);
        cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
        cv::morphologyEx( blurred_image, final_image, cv::MORPH_CLOSE, structuringElement );
        
        t = ros::Time::now();
        string frame_id="camera";
        pub_img.publish(imageToROSmsg(final_image, sensor_msgs::image_encodings::BGR8, frame_id, t));
    }
        loop_rate.sleep();
        
        } 

    return 0;
}