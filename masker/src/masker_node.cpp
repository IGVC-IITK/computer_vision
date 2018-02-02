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
int superpixels=1600;
int N=40;
int img_size=800;
int flag=0;

int mask2d[40][40]={0};
//int mask[576]={0};
int mask[1600]={0};
int labels[40*40]={0};    

void labelsCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
{
	flag=1;
    for(int i=0;i<img_size*img_size;i++)
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
    while(ros::ok())
    {
    	int red_sum[superpixels]={0},green_sum[superpixels]={0},blue_sum[superpixels]={0};

    	ros::spinOnce();
    	if(flag==0)continue;
    	for(int i=0;i<superpixels;i++)
    	{
    	        red_sum[i] =mask[i]*255;
    	        green_sum[i] =mask[i]*255;
    	        blue_sum[i] =mask[i]*255;
    	}
    	
    	cv::Mat M1(img_size,img_size, CV_8UC3, {0,0,0});
    	for (int x=0;x<img_size;x++)
    	{
    	    for(int y=0;y<img_size;y++)
    	    {
    	            M1.at<cv::Vec3b>(x,y)[0]=blue_sum[labels[x*img_size + y ]] ;// b
    	            M1.at<cv::Vec3b>(x,y)[1]=green_sum[labels[x*img_size + y ]] ;// g
    	            M1.at<cv::Vec3b>(x,y)[2]=red_sum[labels[x*img_size + y ]] ;// r
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
