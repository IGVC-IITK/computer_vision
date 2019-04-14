#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sensor_msgs/Imu.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#define SUFF_DELAY 0.15 //calibrated delay between data arrival from image and IMU
#define IMU_FREQ 50.0

int reading_delay = (int) (SUFF_DELAY*IMU_FREQ); //go back these many IMU readings while calculating transform_gimbal
int buffer_size = (int) (SUFF_DELAY*IMU_FREQ*2); //*2 for avoiding overflow
int buffer_end = 0;
double theta_buffer[(int) (SUFF_DELAY*IMU_FREQ*2)];

void getTransform(const sensor_msgs::Imu &Imu)
{
	tf2::Matrix3x3 imu_orient(tf2::Quaternion(Imu.orientation.x, Imu.orientation.y, Imu.orientation.z, Imu.orientation.w));
	double roll, pitch, yaw;
	imu_orient.getRPY(roll, pitch, yaw, 1);
	theta_buffer[buffer_end] = pitch;
	buffer_end = (buffer_end+1)%buffer_size;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg, cv::Mat& birds_image, cv::Size& birds_size, cv::Mat& transform)
{
	std::string image_format;
	if (msg->encoding == sensor_msgs::image_encodings::BGR8)
		image_format = "8UC3";
	else if (msg->encoding == sensor_msgs::image_encodings::MONO8)
		image_format = "8UC1";

	try
	{
		cv::Mat perspective_image=cv_bridge::toCvShare(msg, image_format)->image;
		if(!perspective_image.empty())
		{
			cv::warpPerspective(perspective_image, birds_image, transform, birds_size, 
				CV_INTER_LINEAR | CV_WARP_INVERSE_MAP | CV_WARP_FILL_OUTLIERS);
		}
		cv::waitKey(1);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to '%s'.", msg->encoding.c_str(), image_format.c_str());
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "top_view_node");
	ros::NodeHandle nh;

	// Camera calibration parameters (in pixels)
	// (Currently using the ones for ZED at 720p)
	// double fx = 699.948, fy = 699.948, cx = 629.026, cy = 388.817;
	// (Currently using the ones for Logitech C615)
	double fx = 635.390503, fy = 631.630005, cx = 310.090906, cy = 321.387747;

	// Camera position (in metres, degrees)
	// Note that angle is not important if IMU is used
	double H = 1.5, theta = 40.00;
	theta *= (M_PI/180.0);

	// Defining desired field-of-view (in metres)
	// Ox and Oy are the position of the projection of the optical center on the 
	// ground plane from (0, 0) of the top-view image
	// Wx and Wy are the width and height of the top-view image
	// x-axis is the direction pointing right in the top-view image
	// y-axis is the direction pointing down in the top-view image
	double Ox = 3.20, Oy = 6.30, Wx = 6.40, Wy = 4.80;

	// Scaling factor (in pixels/m)
	// (Use 80.0 for realtime low-res output but 160.0 for datasets)
	double s = 100.0;

	cv::Mat transform(3, 3, CV_64FC1);	
	cv::Size birds_size(s*Wx, s*Wy);
	cv::Mat birds_image(birds_size, CV_8UC1);
	sensor_msgs::ImagePtr top_view_msg;
	image_transport::ImageTransport it_tv(nh);
	image_transport::Subscriber sub_img = it_tv.subscribe("/image", 1, 
		boost::bind(imageCallback, _1, boost::ref(birds_image), boost::ref(birds_size), boost::ref(transform)));
	ros::Subscriber sub_imu = nh.subscribe("/imu", 2000, getTransform);
	image_transport::Publisher pub_tv = it_tv.advertise("/top_view", 1);
	
	ros::Rate loop_rate(50);
	while(nh.ok())
	{
		ros::spinOnce();
		int buffer_pointer = buffer_end - reading_delay;
		if (buffer_pointer < 0)
			buffer_pointer = buffer_pointer + buffer_size;
		//theta = theta_buffer[buffer_pointer];
		ROS_INFO_STREAM("theta = "<<theta);

		// Calculating transformation matrix analytically
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
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				transform.at<double>(i, j) = 
					transform.at<double>(i, j)/transform.at<double>(2, 2);
			}
		ROS_INFO_STREAM("Transformation Matrix:\n"<<transform);

		std::string image_format;
		if (birds_image.type() == CV_8UC3)
			image_format = sensor_msgs::image_encodings::BGR8;
		else if (birds_image.type() == CV_8UC1)
			image_format = sensor_msgs::image_encodings::MONO8;
		top_view_msg = cv_bridge::CvImage(std_msgs::Header(), image_format.c_str(), birds_image).toImageMsg();

		pub_tv.publish(top_view_msg);

		loop_rate.sleep();
	}

	return 0;
}
