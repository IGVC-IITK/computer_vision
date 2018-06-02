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
int superpixels=32*18;
int w=32;
int h=18; // ??? 32 X 18 chahiye
int img_sizex=640;
int img_sizey=360; // 640 X 360 chahiye
int flag=0;

int mask2d[32][18]={0};
//int mask[576]={0};
int mask[32*18]={0};
int labels[640*360]={0};    



void labelsCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
{
	flag=1;
    for(int i=0;i<img_sizex*img_sizey;i++)
        labels[i]=msg->data[i];   
}
void predictionsCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
{
	flag=1;
  for(int i=0;i<superpixels;i++)
    mask[i]=msg->data[i];
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


int main(int argc,char **argv)
{
    ros::init(argc, argv, "masker_node");
    ros::NodeHandle n2;
    ros::Subscriber sub = n2.subscribe("predictions", 1000, predictionsCallback);       
    ros::Subscriber sub2 = n2.subscribe("/gslicr/segmentation", 1000, labelsCallback);       
    image_transport::ImageTransport it_gslicr(n2);
    ros::Time t;
    image_transport::Publisher pub_avg = it_gslicr.advertise("/final_image", 1);
    // get mask from callback function and averaged image (27X27)
    ros::Rate loop_rate(30);
    	int red_sum[superpixels]={0},green_sum[superpixels]={0},blue_sum[superpixels]={0};
    while(ros::ok())
    {
    	ros::spinOnce();
    	for(int i=0;i<superpixels;i++)
    	{
    	        red_sum[i] =mask[i]*255;
    	        green_sum[i] =mask[i]*255;
    	        blue_sum[i] =mask[i]*255;
    	}
    	
    	cv::Mat M1;
        M1.create(cv::Size(640,360), CV_8UC3);
    	for (int j=0;j<img_sizey;j++)
    	{
    	    for(int i=0;i<img_sizex;i++)
    	    {
    	            M1.at<cv::Vec3b>(j,i)[0]=blue_sum[labels[j*img_sizex + i ]] ;// b
    	            M1.at<cv::Vec3b>(j,i)[1]=green_sum[labels[j*img_sizex + i ]] ;// g
    	            M1.at<cv::Vec3b>(j,i)[2]=red_sum[labels[j*img_sizex + i ]] ;// r
    	    }
    	}
    	//cv::namedWindow("video",1);
    	//cv::imgshow("video",M1);
    	string frame_id="camera";
    	t = ros::Time::now();
    	pub_avg.publish(imageToROSmsg(M1, sensor_msgs::image_encodings::BGR8, frame_id, t));
    	loop_rate.sleep();

	}
 return 0;
}
