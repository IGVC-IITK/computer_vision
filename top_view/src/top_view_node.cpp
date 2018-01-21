#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>

using namespace std;
cv::Mat Z(3,3,CV_64FC1);
cv::Mat birds_image;

void msgRecieved(const sensor_msgs::ImageConstPtr& msg)
{
  try
    {
      cv::Mat gs=cv_bridge::toCvShare(msg, "bgr8")->image;

      //cv::namedWindow("IMAGE",1);
      //cv::namedWindow("original",1);
      if(!gs.empty())
      {
        cv::warpPerspective( gs, birds_image, Z, cv::Size( 1280, 1280 ) ,
                         CV_INTER_LINEAR | CV_WARP_INVERSE_MAP | CV_WARP_FILL_OUTLIERS ) ;
        cout<<Z<<endl;
      }
        //cv::imshow("IMAGE",birds_image);
        //cv::imshow("original",gs);
        cv::waitKey(30);

    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc,char** argv)
{ 
  ros::init(argc,argv,"reciever");
  cout<< argc << "  " << **argv <<endl;
  ros::NodeHandle n;

  ifstream fs("/home/bhatti/catkin_ws/top_view.txt");
  char c;
  fs>>c;
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        fs>>Z.at<double>(i,j);
        fs>>c;
      }
    }
    fs.close();
  cout<<Z<<endl;
  image_transport::ImageTransport it(n);
        image_transport::Publisher pub = it.advertise("/top_view", 1);
    ros::Rate loop_rate(30);
    image_transport::Subscriber sub = it.subscribe("/obstacle_free_image",1,msgRecieved);
    while(n.ok()){
  


        
  
        sensor_msgs::ImagePtr msg1;
              msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", birds_image).toImageMsg();
  pub.publish(msg1);
  ros::spinOnce();
       // cout<<msg1<<endl;

loop_rate.sleep();
}

  return 0;
}