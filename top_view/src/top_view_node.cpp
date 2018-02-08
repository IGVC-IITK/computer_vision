#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void imageCallback(const sensor_msgs::ImageConstPtr& msg, cv::Mat& birds_image, cv::Size& birds_size, cv::Mat& transform)
{
	try
	{
		cv::Mat perspective_image=cv_bridge::toCvShare(msg, "bgr8")->image;
		if(!perspective_image.empty())
		{
			cv::warpPerspective(perspective_image, birds_image, transform, birds_size, 
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

	// Camera calibration parameters (in pixels)
	double fx = 699.948, fy = 699.948, cx = 629.026, cy = 388.817;
	// Camera position (in metres, degrees)
	double H = 0.75, theta = 10.00;
	theta *= (M_PI/180 );
	// Defining desired field-of-view (in metres)
	double Ox = 5.00, Oy = 5.00, Wx = 10.00, Wy = 10.00;
	// Scaling factor (in pixels/m)
	double s = 128.00;

	// Calculating transformation matrix analytically
	cv ::Mat transform(3, 3, CV_64FC1);

	transform.at<double>(0, 0) = fx;
	transform.at<double>(0, 1) = -cx*cos(theta);
	transform.at<double>(0, 2) = s*(cx*(H*sin(theta)+Oy*cos(theta)) - Ox*fx);

	transform.at<double>(1, 0) = 0;
	transform.at<double>(1, 1) = fy*sin(theta) - cy*cos(theta);
	transform.at<double>(1, 2) = s*(cy*(H*sin(theta)+Oy*cos(theta)) + 
									fy*(H*cos(theta)-Oy*sin(theta)));

	transform.at<double>(2, 0) = 0;
	transform.at<double>(2, 1) = -cos(theta);
	transform.at<double>(2, 2) = s*(H*sin(theta) + Oy*cos(theta));

	// Normalizing transformation matrix
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
		{
			transform.at<double>(i, j) = 
				transform.at<double>(i, j)/transform.at<double>(2, 2);
		}
	ROS_INFO_STREAM("Transformation Matrix:\n"<<transform);
	
	cv::Size birds_size(s*Wx, s*Wy);
	cv::Mat birds_image(birds_size, CV_8UC3);
	sensor_msgs::ImagePtr top_view_msg;
	image_transport::ImageTransport it_tv(n);
	image_transport::Subscriber sub = it_tv.subscribe("/image", 1, 
		boost::bind(imageCallback, _1, boost::ref(birds_image), boost::ref(birds_size), boost::ref(transform)));
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
