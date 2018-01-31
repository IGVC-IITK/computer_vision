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

int inner_superpixels = 576; //
int found_total_superpixels = 729; // THESE values are of no use , we are actually getting parameters from node handle
int N=26;  // 

 /*
int label[640*480]={0};
 
 void messageCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
 {
  for(int i=0;i<inner_superpixels;i++)
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
 	  n.getParam("inner_superpixels",inner_superpixels);
 	  n.getParam("found_total_superpixels",found_total_superpixels);
 	  n.getParam("N",N);

 	  ros::ServiceClient master = n.serviceClient<classifier::lane_classifier>("classifier");
 	  classifier::lane_classifier srv;
 	  srv.request.data.clear();
    ros::Publisher pub= n.advertise<std_msgs::UInt16MultiArray>("predictions",1000);

 	// ==========================================================


 	  clock_t t;
    t = clock();
    int mask[found_total_superpixels];
 	
  //==========================================================
    
    //ros::Subscriber sub=n.subscribe("/gslicr/segmentation", 1000, messageCallback);

    cv::Mat image;
    image.create(cv::Size(N, N), CV_8UC3);
      image_transport::ImageTransport it_avg(n);
      image_transport::Subscriber sub_img = it_avg.subscribe("/gslicr/averages", 1, boost::bind(imageCallback, _1, boost::ref(image)));
    while(ros::ok())
    {

  //=============================================================
      ros::spinOnce();
//    for(int i=0;i<27 * inner_superpixels;i++)
  //    srv.request.data.push_back(1); // load all data 
    cv::Size sz=image.size();
    int h=sz.height;
    int w=sz.width;
    cout<<h<<" "<<w<<endl;
    //cv::namedWindow("abc",1);
    //cv::imshow("abc",image);
    int red[h][w],green[h][w],blue[h][w];
    for(int i=0;i<h;i++)
    {
      for(int j=0;j<w;j++)
      {
        blue[i][j]=(int)image.at<cv::Vec3b>(i,j)[0];
        green[i][j]=(int)image.at<cv::Vec3b>(i,j)[1];
        red[i][j]=(int)image.at<cv::Vec3b>(i,j)[2];
      }
    }    
    int cnt =0;
    for(int i=0;i<h;i++)
    {
      for(int j=0;j<w;j++)
      {
        if(i==0||j==0||i==h-1||j==w-1)
          continue;
        srv.request.data.push_back(red[i-1][j-1]);
        srv.request.data.push_back(green[i-1][j-1]);
        srv.request.data.push_back(blue[i-1][j-1]);
        srv.request.data.push_back(red[i-1][j]);
        srv.request.data.push_back(green[i-1][j]);
        srv.request.data.push_back(blue[i-1][j]);
        srv.request.data.push_back(red[i-1][j+1]);
        srv.request.data.push_back(green[i-1][j+1]);
        srv.request.data.push_back(blue[i-1][j+1]);
        srv.request.data.push_back(red[i][j-1]);
        srv.request.data.push_back(green[i][j-1]);
        srv.request.data.push_back(blue[i][j-1]);
        srv.request.data.push_back(red[i][j]);
        srv.request.data.push_back(green[i][j]);
        srv.request.data.push_back(blue[i][j]);
        srv.request.data.push_back(red[i][j+1]);
        srv.request.data.push_back(green[i][j+1]);
        srv.request.data.push_back(blue[i][j+1]);
        srv.request.data.push_back(red[i+1][j-1]);
        srv.request.data.push_back(green[i+1][j-1]);
        srv.request.data.push_back(blue[i+1][j-1]);
        srv.request.data.push_back(red[i+1][j]);
        srv.request.data.push_back(green[i+1][j]);
        srv.request.data.push_back(blue[i+1][j]);
        srv.request.data.push_back(red[i+1][j+1]);
        srv.request.data.push_back(green[i+1][j+1]);
        srv.request.data.push_back(blue[i+1][j+1]);
        cnt++;
      }
    }
  //===========================================================
    cout<<cnt<<endl;
    if (master.call(srv))
    {
      for(int i=0;i<inner_superpixels;i++)
        mask[i]=(int)srv.response.ans[i]; // get whole mask
    }
    else
    {
      ROS_ERROR("Failed to call service classifier");
      return 1;
    }


    ros::Rate loop_rate(10);
    
    int count=0;
    
      std_msgs::UInt16MultiArray msg;
      for(int i=0;i<inner_superpixels;i++)
      {
        msg.data.push_back(mask[i]);
      }
      pub.publish(msg);
      msg.data.clear();
      srv.request.data.clear();
      
      loop_rate.sleep();
      ++count;
    }

   	for (int i=0;i<found_total_superpixels;i++)cout<<mask[i]<<" "; // prints the mask

   	srv.request.data.clear();

   	cout<<endl;
   	t = clock() - t;
   	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
   	cout<<(1/time_taken)<<" FPS of prediction"<<endl;
    //ros::spin();
    return 0;
 }
