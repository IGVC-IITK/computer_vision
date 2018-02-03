#include "ros/ros.h"
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
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>


using namespace :: std;
using namespace cv;
using namespace cv::ml;


   
int flag=0;

void imageCallback(const sensor_msgs::ImageConstPtr& imgMessage, cv::Mat& image)
{
  	flag=1;
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




int main (int argc, char **argv)
{
    int inner_superpixels = 40*40; 
    int found_total_superpixels = 40*40;
    int N=40;

    ros::init(argc, argv, "classifier_client");
    ros::NodeHandle n;

    //getting ros parameters
    n.getParam("inner_superpixels",inner_superpixels);
    n.getParam("found_total_superpixels",found_total_superpixels);
    n.getParam("N",N);

    ros::Publisher pub= n.advertise<std_msgs::UInt16MultiArray>("predictions",1000);
	
    vector<int> mask(found_total_superpixels,4);
    cv::Mat image;
    image.create(cv::Size(N, N), CV_8UC3);
    image_transport::ImageTransport it_avg(n);
    image_transport::Subscriber sub_img = it_avg.subscribe("/gslicr/averages", 1, boost::bind(imageCallback, _1, boost::ref(image)));

    int g=0;
    int red[40][40],green[40][40],blue[40][40];
    while(ros::ok())
    {

    ros::spinOnce();
    if(flag==0)
    	continue;
    
    cv::Size sz=image.size();
    int h=sz.height;
    int w=sz.width;

    for(int i=0;i<40;i++)
    {
      for(int j=0;j<40;j++)
      {
        blue[i][j]=(int)image.at<cv::Vec3b>(i,j)[0];
        green[i][j]=(int)image.at<cv::Vec3b>(i,j)[1];
        red[i][j]=(int)image.at<cv::Vec3b>(i,j)[2];
      }
    }

    int lines=h*w;
    int features=75;
    Mat_<float> data(lines,features);
    Mat_<float> result(lines,1);

    int x[]={-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2};
    int y[]={-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2};
    
    for(int i=0;i<lines;i++)
    {
        for(int j=0;j<25;j++)
        {
            int i1=i/w + x[j];
            int j1=i%w + y[j];

            if(i1==-1 || i1==-2 || j1==-1 || j1==-2 || i1==h || i1==h+1 || j1==w || j1==w+1)
            {
                data(i,3*j)=100;
                data(i,3*j+1)=50;
                data(i,3*j+2)=150;
                continue;
                    
            }

            data(i,3*j)=red[i/w+x[j]][i%w+y[j]];
            data(i,3*j+1)=green[i/w+x[j]][i%w+y[j]];
            data(i,3*j+2)=blue[i/w+x[j]][i%w+y[j]];

        }
    }
    cout<<data(0,0)<<endl;

    Ptr<ANN_MLP> network = cv::ml::ANN_MLP::load("/home/utkarsh/Downloads/mlp.yml");    
    //cout<<h<<" "<<w<<endl;
    network->predict(data,result);
    //cout<<"doo"<<endl;
    //cout<<h<<" "<<w<<endl;
    if (network->isTrained())
    {
        for (int i=0; i<data.rows; ++i)
        {
            if(result(i,0)>0)
                mask[i]=1;
            else
                mask[i]=0; 
      		cout<<mask[i];//<<"doo";
        }
    }
    //cout<<endl;
    
    ros::Rate loop_rate(10);

    std_msgs::UInt16MultiArray msg;
    for(int i=0;i<inner_superpixels;i++){
      msg.data.push_back(mask[i]);
    }

    pub.publish(msg);
    msg.data.clear();
    //loop_rate.sleep();
    }

  //  for (int i=0;i<found_total_superpixels;i++)cout<<mask[i]<<" "; // prints the mask

return 0;
}
