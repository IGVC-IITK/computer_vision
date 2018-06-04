 #include "ros/ros.h"
 #include "classifier/lane_classifier.h"
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
 using namespace :: std;
 /*
int label[640*480]={0};
 
 void messageCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
 {
  for(int i=0;i<576;i++)
  {
    label[i]=msg->data[i];
  }
  return;
 }
*/
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


 int main(int argc, char **argv)
 {
 	  ros::init(argc, argv, "classifier_client");
 	/*  if (argc != 3)
 	  {
 	    ROS_INFO("usage: classifier_client");
 	    return 1;
 	  }
*/
 	  ros::NodeHandle n;
      ros::Publisher pub= n.advertise<std_msgs::UInt16MultiArray>("predictions",1000);

 	// ==========================================================


 	  clock_t t;
    t = clock();
 	
  //==========================================================
    
    //ros::Subscriber sub=n.subscribe("/gslicr/segmentation", 1000, messageCallback);

    cv::Mat image,hsv,out;
    image.create(cv::Size(640, 360), CV_8UC3); // previously it was 40 X 40 
      image_transport::ImageTransport it_avg(n);
      image_transport::Subscriber sub_img = it_avg.subscribe("top_view", 1, boost::bind(imageCallback, _1, boost::ref(image)));
      ros::Rate loop_rate(10);
     
      cv::namedWindow("ge");
      int h1 = 150;
     cv::createTrackbar("h1","ge",&h1,255);
	 int s1 = 150;
     cv::createTrackbar("s1","ge",&s1,255);
      int v1 = 150;
     cv::createTrackbar("v1","ge",&v1,255);

       int h2 = 150;
     cv::createTrackbar("h2","ge",&h2,255);
	 int s2 = 150;
     cv::createTrackbar("s2","ge",&s2,255);
      int v2 = 150;
     cv::createTrackbar("v2","ge",&v2,255);


    while(ros::ok())
    {
    	
    	cv::waitKey(1);
    	cv::GaussianBlur(image, image, cv::Size(9,9), 5); // tune
    	cv::Mat channel[3],alt;
    	
    	//temp = image(Rect(0,0))
    	cv::split(image,channel);
    	//alt = 2*channel[1] - channel[0];
    	alt = image;
		//alt = (channel[2] > r1); alt = ( channel[2] <r2);
		//alt = (channel[1] > g1); alt = ( channel[1] <g2);
		//alt = (channel[0] > b1); alt = ( channel[0] <b2);

    	cv::cvtColor(image,hsv, CV_BGR2HSV);
    	cv::Scalar colorlow(h1,s1,v1);
    	cv::Scalar colorhigh(h2,s2,v2);
    	
    	cv::inRange(hsv, colorhigh, colorlow ,out);

    	char key = cv::waitKey(26);
    	if(key == 'q')
    	{
    		fstream f;
    		f.open("/home/bhatti/abhishek/params.txt");
    		f<<h1<<" "<<s1<<" "<<v1<<" "<<h2<<" "<<s2<<" "<<v2;
    	}


		cv::imshow("fg",out);


    	ros::spinOnce();
    }
   
    return 0;
 }
