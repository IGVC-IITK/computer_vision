#include <ros/ros.h>
#include <ros/package.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>

void imageCallback(const sensor_msgs::ImageConstPtr& msg, cv::Mat& birds_image, cv::Mat& transform)
{
	try
	{
		cv::Mat perspective_image=cv_bridge::toCvShare(msg, "bgr8")->image;
		if(!perspective_image.empty())
		{
			cv::warpPerspective(perspective_image, birds_image, transform, cv::Size( 1406, 720 ), 
				CV_INTER_LINEAR | CV_WARP_INVERSE_MAP | CV_WARP_FILL_OUTLIERS);
		}
		cv::waitKey(1);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "top_view_node");
	ros::NodeHandle n;

	// Reading transformation matrix from file
	std::string package_path = ros::package::getPath("top_view");
	std::string file_path = package_path+"/src/top_view.txt";
	std::ifstream fs(file_path.c_str());
	cv::Mat transform(3, 3, CV_64FC1);
	char c;
	fs>>c;
	for (int i = 0; i < 3; ++i)
	{
			for (int j = 0; j < 3; ++j)
			{
					fs>>transform.at<double>(i,j);
					fs>>c;
			}
	}
	fs.close();
	std::cout<<transform<<std::endl;
	
	cv::Mat birds_image(640, 480, CV_8UC3);
	sensor_msgs::ImagePtr top_view_msg;
	image_transport::ImageTransport it_tv(n);
	image_transport::Subscriber sub = it_tv.subscribe("/image", 1, 
		boost::bind(imageCallback, _1, boost::ref(birds_image), boost::ref(transform)));
	image_transport::Publisher pub = it_tv.advertise("/top_view", 1);
	
	ros::Rate loop_rate(10);
	while(n.ok())
	{
		top_view_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", birds_image).toImageMsg();
		pub.publish(top_view_msg);
		ros::spinOnce();

		loop_rate.sleep();
	}

	return 0;
}
