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

    /*std::string line;
    ifstream myfile ("/home/abhishek/params.txt");
    if (myfile.is_open())
    {
      while ( getline (myfile,line) )
      {
        ROS_INFO("%s",line);
      }
      myfile.close();
    }*/
    int h1=143;
    int s1=41;
    int v1=146;
    int h2=73;
    int s2=0;
    int v2=66;
    /*fstream f;
    f.open("/home/bhatti/abhishek/params.txt");
    f>>h1;
    f>>s1;
    f>>v1;
    f>>h1;
    f>>s2;
    f>>v2;
    ROS_INFO("%d",s1);*/
    
    

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
   ros::Time t;
  //==========================================================
    
    //ros::Subscriber sub=n.subscribe("/gslicr/segmentation", 1000, messageCallback);

    cv::Mat image,hsv,out,rec_out,im_with_keypoints;
    image.create(cv::Size(640, 360), CV_8UC3); // previously it was 40 X 40 
      image_transport::ImageTransport it_avg(n);
      image_transport::Subscriber sub_img = it_avg.subscribe("top_view", 1, boost::bind(imageCallback, _1, boost::ref(image)));
      image_transport::Publisher pub_avg = it_avg.advertise("/final_image", 1);

      ros::Rate loop_rate(10);

     

      std::vector <cv::Vec4i> hierarchy;
      std::vector <std::vector<cv::Point> > contours;


    while(ros::ok())
    {
    	if(!image.empty()){
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

		//cv::imshow("fg",out);

    //cv::SimpleBlobDetector detector;
    cv::SimpleBlobDetector::Params params;
    //params.minThreshold = 10;
    //params.maxThreshold = 200;
     
    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 100;
 


    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params); 
    
    std::vector<cv::KeyPoint> keypoints;



    detector->detect( out, keypoints);
    
    cv::drawKeypoints( out, keypoints, im_with_keypoints, cv::Scalar(0,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    //cv::line(im_with_keypoints, cv::Point(0,0), cv::Point(100,100), cv::Scalar(0,0,255));
    

    cv::resize(im_with_keypoints, im_with_keypoints, cv::Size(160,90));
    //cv::imshow("before", im_with_keypoints );// Show blobs

    im_with_keypoints = (im_with_keypoints>0);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,1));

    cv::erode(im_with_keypoints,im_with_keypoints,element,cv::Point(-1,-1),1);



    //cv::imshow("after", im_with_keypoints );// Show blobs

    string frame_id="camera";
      t = ros::Time::now();
      pub_avg.publish(imageToROSmsg(im_with_keypoints, sensor_msgs::image_encodings::BGR8, frame_id, t));
      loop_rate.sleep();
    }

    	ros::spinOnce();
    }
   
    return 0;
 }
