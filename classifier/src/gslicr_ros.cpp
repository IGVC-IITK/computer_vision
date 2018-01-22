// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "ros/ros.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

#include <boost/make_shared.hpp>

using namespace std;

cv::Mat oldFrame;
string img_frame_id = "camera";

void load_image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<cv::Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<cv::Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<cv::Vec3b>(y, x)[2];
		}
}

void load_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<cv::Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<cv::Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<cv::Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}

void imageCallback(const sensor_msgs::ImageConstPtr& imgMessage)
{
	img_frame_id = imgMessage->header.frame_id;
	cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(imgMessage, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    oldFrame = cv_ptr->image;
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

int main(int argc, char **argv)
{
	ros::init(argc, argv, "gslicr_ros_node");
	ros::NodeHandle nh;
	ros::Time t;
	image_transport::ImageTransport it_gslicr(nh);
	image_transport::Subscriber sub_img = it_gslicr.subscribe("camera/image_raw", 1, imageCallback);
	image_transport::Publisher pub_seg = it_gslicr.advertise("gslicr/segmentation", 1);

	// gSLICr settings
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = 640;
	my_settings.img_size.y = 480;
	my_settings.no_segs = 750;
	my_settings.spixel_size = 16;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

	// gSLICr takes gSLICr::UChar4Image as input and out put
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	cv::Size s(my_settings.img_size.x, my_settings.img_size.y);
	cv::Mat frame;
	cv::Size inputSize(640, 480);
	oldFrame.create(inputSize, CV_8UC3);
	cv::Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);

	StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
	
	ros::Rate loop_rate(60);
	while (ros::ok())
	{
		ros::spinOnce();
		cv::resize(oldFrame, frame, s);
		
		load_image(frame, in_img);
		sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
		gSLICr_engine->Process_Frame(in_img);
		sdkStopTimer(&my_timer); 
		// cout<<"\rsegmentation in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
		
		gSLICr_engine->Draw_Segmentation_Result(out_img);
		load_image(out_img, boundry_draw_frame);
		t = ros::Time::now();
		pub_seg.publish(imageToROSmsg(boundry_draw_frame, sensor_msgs::image_encodings::BGR8, img_frame_id, t));

		loop_rate.sleep();
	}

	return 0;
}
