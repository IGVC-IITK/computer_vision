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
int superpixels=1600;
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
 	  ros::ServiceClient master = n.serviceClient<classifier::lane_classifier>("classifier");
 	  classifier::lane_classifier srv;
 	  srv.request.data.clear();
    ros::Publisher pub= n.advertise<std_msgs::UInt16MultiArray>("predictions",1000);

 	// ==========================================================


 	  clock_t t;
    t = clock();
    int mask[superpixels];
 	
  //==========================================================
    
    //ros::Subscriber sub=n.subscribe("/gslicr/segmentation", 1000, messageCallback);

    cv::Mat image;
    image.create(cv::Size(40, 40), CV_8UC3);
      image_transport::ImageTransport it_avg(n);
      image_transport::Subscriber sub_img = it_avg.subscribe("/gslicr/averages", 1, boost::bind(imageCallback, _1, boost::ref(image)));
      ros::Rate loop_rate(10);
    while(ros::ok())
    {

  //=============================================================
      ros::spinOnce();
//    for(int i=0;i<27 * 576;i++)
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
       for(int i1=-2;i1<=2;i1++)
        {
          for(int j1=-2;j1<=2;j1++){

            if(i+i1==-1 || i+i1==-2 || j+j1==-1 || j+j1==-2 || i+i1==h+1 || i+i1==h || j+j1==w || j+j1==w+1)
            {
                 srv.request.data.push_back(100);
                 srv.request.data.push_back(150);
                 srv.request.data.push_back(50);
                continue;    
            }
        srv.request.data.push_back(red[i+i1][j+j1]);
        srv.request.data.push_back(green[i+i1][j+j1]);
        srv.request.data.push_back(blue[i+i1][j+j1]);
        
      }
    }
        cnt++;
      }
    }
   

  //===========================================================
    cout<<cnt<<endl;
    if (master.call(srv))
    {
      for(int i=0;i<superpixels;i++)
        mask[i]=(int)srv.response.ans[i]; // get whole mask
    }
    else
    {
      ROS_ERROR("Failed to call service classifier");
      return 1;
    }
    
    int count=0;
    
      std_msgs::UInt16MultiArray msg;
      for(int i=0;i<superpixels;i++)
      {
        msg.data.push_back(mask[i]);
      }
      pub.publish(msg);
      msg.data.clear();
      srv.request.data.clear();
      
      loop_rate.sleep();
      ++count;
    }

   	for (int i=0;i<superpixels;i++)cout<<mask[i]<<" "; // prints the mask

   	srv.request.data.clear();

   	cout<<endl;
   	t = clock() - t;
   	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
   	cout<<(1/time_taken)<<" FPS of prediction"<<endl;
    //ros::spin();
    return 0;
 }
