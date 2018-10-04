// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>
#include <stdint.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "ros/ros.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/UInt16MultiArray.h>

#include <boost/make_shared.hpp>

using namespace std;

void imageCallback(const sensor_msgs::ImageConstPtr& imgMessage, cv::Mat& inimg, gSLICr::UChar4Image* outimg, cv::Size s, string& frame_id)
{
	frame_id = imgMessage->header.frame_id;	// for assigning the correct frame_id to the published images
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
	cv::resize(cv_ptr->image, inimg, s);

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

	// gSLICr settings
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = 640;
	my_settings.img_size.y = 360;
	my_settings.no_segs = 32*18;
	my_settings.spixel_size = 20;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::CIELAB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

	// gSLICr takes gSLICr::UChar4Image as input and output
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	cv::Size outputSize(my_settings.img_size.x, my_settings.img_size.y);
	int spixel_size;
	if (my_settings.seg_method == gSLICr::GIVEN_NUM)
	{
		float cluster_size = (float)(my_settings.img_size.x * my_settings.img_size.y) / (float)my_settings.no_segs;
		spixel_size = (int)ceil(sqrtf(cluster_size));
	}
	else
	{
		spixel_size = my_settings.spixel_size;
	}
	int spixels_x = (int)ceil(my_settings.img_size.x / spixel_size);
	int spixels_y = (int)ceil(my_settings.img_size.y / spixel_size);
	cv::Size gridSize(spixels_x, spixels_y);
	string frame_id = "camera";
	cv::Mat frame, boundary_draw_frame, sum_frame, count_frame, average_frame, average_rpl_frame;
	frame.create(outputSize, CV_8UC3);
	boundary_draw_frame.create(outputSize, CV_8UC3);
	sum_frame.create(gridSize, CV_32SC3);
	count_frame.create(gridSize, CV_16UC1);
	average_frame.create(gridSize, CV_8UC3);
	average_rpl_frame.create(outputSize, CV_8UC3);
	StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
	std_msgs::UInt16MultiArray labels;	
	labels.data.resize(my_settings.img_size.x*my_settings.img_size.y);

	image_transport::ImageTransport it_gslicr(nh);
	image_transport::Subscriber sub_img = it_gslicr.subscribe("/top_view", 1, boost::bind(imageCallback, _1, boost::ref(frame), in_img, boost::ref(outputSize), boost::ref(frame_id)));
	image_transport::Publisher pub_bd = it_gslicr.advertise("/gslicr/boundaries", 1);
	ros::Publisher pub_seg = nh.advertise<std_msgs::UInt16MultiArray>("/gslicr/segmentation", 1);
	// image_transport::Publisher pub_resize = it_gslicr.advertise("/gslicr/resize", 1);	// for comparing with cv::resize
	image_transport::Publisher pub_avg = it_gslicr.advertise("/gslicr/averages", 1);
	image_transport::Publisher pub_avg_rpl = it_gslicr.advertise("/gslicr/average_replaced_image", 1);
	
	ros::Rate loop_rate(60);
	while (ros::ok())
	{
		ros::spinOnce();

		// main computation
		if(pub_bd.getNumSubscribers() + pub_seg.getNumSubscribers() + pub_avg.getNumSubscribers() + pub_avg_rpl.getNumSubscribers() > 0)
		{
			sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
			gSLICr_engine->Process_Frame(in_img);
			sdkStopTimer(&my_timer); 
			// cout<<"\rsegmentation in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
		}

		// boundaries
		if(pub_bd.getNumSubscribers() > 0)
		{
			gSLICr_engine->Draw_Segmentation_Result(out_img);
			load_image(out_img, boundary_draw_frame);
			t = ros::Time::now();
			pub_bd.publish(imageToROSmsg(boundary_draw_frame, sensor_msgs::image_encodings::BGR8, frame_id, t));
		}

		// segmentation labels
		if(pub_seg.getNumSubscribers() + pub_avg.getNumSubscribers() + pub_avg_rpl.getNumSubscribers() > 0)
			gSLICr_engine->Write_Seg_Res_To_Array(labels.data);
		if(pub_seg.getNumSubscribers() > 0)
			pub_seg.publish(labels);

		// cv::resize(frame, average_frame, gridSize);
		// pub_resize.publish(imageToROSmsg(average_frame, sensor_msgs::image_encodings::BGR8, frame_id, t));

		// average colour values
		if(pub_avg.getNumSubscribers() + pub_avg_rpl.getNumSubscribers() > 0)
		{
			// calculating the sum of the pixel values for each segment
			sum_frame = cv::Mat::zeros(gridSize, CV_32SC3);
			count_frame = cv::Mat::zeros(gridSize, CV_16UC1);
			for(int i=0; i<my_settings.img_size.x*my_settings.img_size.y; i++)
			{
				sum_frame.at<cv::Vec3i>(labels.data[i]/spixels_x, labels.data[i]%spixels_x)[0] += frame.at<cv::Vec3b>(i/my_settings.img_size.x, i%my_settings.img_size.x)[0];
				sum_frame.at<cv::Vec3i>(labels.data[i]/spixels_x, labels.data[i]%spixels_x)[1] += frame.at<cv::Vec3b>(i/my_settings.img_size.x, i%my_settings.img_size.x)[1];
				sum_frame.at<cv::Vec3i>(labels.data[i]/spixels_x, labels.data[i]%spixels_x)[2] += frame.at<cv::Vec3b>(i/my_settings.img_size.x, i%my_settings.img_size.x)[2];
				count_frame.at<ushort>(labels.data[i]/spixels_x, labels.data[i]%spixels_x) += 1;
			}
			// calculating average colour values for each superpixel
			for(int j=0; j<spixels_y; j++)
				for(int i=0; i<spixels_x; i++)
				{
					if(count_frame.at<ushort>(j, i) != 0)
					{
						// black to green
						average_frame.at<cv::Vec3b>(j, i)[0] = sum_frame.at<cv::Vec3i>(j, i)[0]/count_frame.at<ushort>(j, i);
						average_frame.at<cv::Vec3b>(j, i)[1] = sum_frame.at<cv::Vec3i>(j, i)[1]/count_frame.at<ushort>(j, i);
						average_frame.at<cv::Vec3b>(j, i)[2] = sum_frame.at<cv::Vec3i>(j, i)[2]/count_frame.at<ushort>(j, i);
						if (average_frame.at<cv::Vec3b>(j, i)[0] < 16 && average_frame.at<cv::Vec3b>(j, i)[1] < 16 && average_frame.at<cv::Vec3b>(j, i)[2] < 16)
						{
							average_frame.at<cv::Vec3b>(j, i)[0] = 0;
							average_frame.at<cv::Vec3b>(j, i)[1] = 0;
							average_frame.at<cv::Vec3b>(j, i)[2] = 0;
						}
					}
					else	// for avoiding divide-by-zero error
					{
						average_frame.at<cv::Vec3b>(j, i)[0] = 0;
						average_frame.at<cv::Vec3b>(j, i)[1] = 0;
						average_frame.at<cv::Vec3b>(j, i)[2] = 0;
					}
				}
		}
		if(pub_avg.getNumSubscribers() > 0)
			pub_avg.publish(imageToROSmsg(average_frame, sensor_msgs::image_encodings::BGR8, frame_id, t));

		// average-replaced-image
		if(pub_avg_rpl.getNumSubscribers() > 0)
		{
			for(int i=0; i<my_settings.img_size.x*my_settings.img_size.y; i++)
			{
				average_rpl_frame.at<cv::Vec3b>(i/my_settings.img_size.x, i%my_settings.img_size.x)[0] = average_frame.at<cv::Vec3b>(labels.data[i]/spixels_x, labels.data[i]%spixels_x)[0];
				average_rpl_frame.at<cv::Vec3b>(i/my_settings.img_size.x, i%my_settings.img_size.x)[1] = average_frame.at<cv::Vec3b>(labels.data[i]/spixels_x, labels.data[i]%spixels_x)[1];
				average_rpl_frame.at<cv::Vec3b>(i/my_settings.img_size.x, i%my_settings.img_size.x)[2] = average_frame.at<cv::Vec3b>(labels.data[i]/spixels_x, labels.data[i]%spixels_x)[2];
			}
			pub_avg_rpl.publish(imageToROSmsg(average_rpl_frame, sensor_msgs::image_encodings::BGR8, frame_id, t));
		}
		
		loop_rate.sleep();
	}

	return 0;
}
