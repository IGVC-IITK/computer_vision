#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <queue>
#include <vector>
#include <iomanip>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/UInt16MultiArray.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
using namespace std;
using namespace cv;

vector <int > splx1;
vector <int > splx2;
vector <pair<int,int> > sply1;
vector <pair<int,int> > sply2;
static int mask2d[24][24];
//int mask[576]={0};
static int mask[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};    
static int labels[500*500]={0};    
void spline(int N, vector<pair<int,int> >& eqn, vector<int>& loop)
{
    int i,j,k,n;
    cout.precision(4);                        //set precision
    cout.setf(ios::fixed);
    double x[N],y[N];
    for(int i=0;i<N;i++){
        x[i]=eqn[i].first;
    }
    for(int i=0;i<N;i++){
        y[i]=eqn[i].second;
    }
    n=3;                                // n is the degree of Polynomial 
    double X[2*n+1];                        //Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    for (i=0;i<2*n+1;i++)
    {
        X[i]=0;
        for (j=0;j<N;j++)
            X[i]=X[i]+pow(x[j],i);        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    }
    double B[n+1][n+2],a[n+1];            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
    for (i=0;i<=n;i++)
        for (j=0;j<=n;j++)
            B[i][j]=X[i+j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
    double Y[n+1];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    for (i=0;i<n+1;i++)
    {    
        Y[i]=0;
        for (j=0;j<N;j++)
        Y[i]=Y[i]+pow(x[j],i)*y[j];        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    }
    for (i=0;i<=n;i++)
        B[i][n+1]=Y[i];                //load the values of Y as the last column of B(Normal Matrix but augmented)
    n=n+1; 
    for (i=0;i<n;i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
        for (k=i+1;k<n;k++)
            if (B[i][i]<B[k][i])
                for (j=0;j<=n;j++)
                {
                    double temp=B[i][j];
                    B[i][j]=B[k][j];
                    B[k][j]=temp;
                }
    
    for (i=0;i<n-1;i++)            //loop to perform the gauss elimination
        for (k=i+1;k<n;k++)
            {
                double t=B[k][i]/B[i][i];
                for (j=0;j<=n;j++)
                    B[k][j]=B[k][j]-t*B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
            }
    for (i=n-1;i>=0;i--)                //back-substitution
    {                        //x is an array whose values correspond to the values of x,y,z..
        a[i]=B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=0;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
                a[i]=a[i]-B[i][j]*a[j];
        a[i]=a[i]/B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }
    for(int k=0;k<loop.size();k++){
        int j,i;
        i=loop[k];
        double jj= a[0] + a[1]*i + a[2]*i*i + a[3]*i*i*i;
        j=int(jj);
        if(mask2d[i][j]==0 && j<24){
            mask2d[i][j]=1;
        }
    }
}

void splfit(int mask[])
{
        splx1.clear();
        sply1.clear();
        splx2.clear();
        sply2.clear();

        for(int i=0;i<24;i++)
            for(int j=0;j<24;j++)
                mask2d[i][j]=mask[i*24+j];
        
        int N,flag1,flag2;
        
        for(int i=0;i<24;i++){
            flag1=0;
            for(int j=0;j<12;j++){
                if(mask2d[i][j]==1){
                    sply1.push_back(make_pair(i,j));
                    flag1=1;
                }
                if(flag1==1)
                    break;
            }
            if(flag1==0)
                splx1.push_back(i);
        }
        for(int i=0;i<24;i++){
            flag2=0;
            for(int j=12;j<24;j++){
                if(mask2d[i][j]==1){
                    sply2.push_back(make_pair(i,j));
                    flag2=1;
                }
                if(flag2==1)
                    break;
            }
            if(flag2==0)
                splx2.push_back(i);
        }
        cout << flush;
        spline(sply1.size(), sply1, splx1);
        spline(sply2.size(), sply2, splx2);

        for(int i=0;i<24;i++)
            for(int j=0;j<24;j++)
                mask[i*24+j]=mask2d[i][j];
} 
void labelsCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
{
    for(int i=0;i<500*500;i++)
        labels[i]=msg->data[i];   
}
void predictionsCallback(const std_msgs::UInt16MultiArray::ConstPtr& msg)
{
  for(int i=0;i<576;i++)
    mask[i]=msg->data[i];
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


int main(int argc,char **argv)
{
    ros::init(argc, argv, "masker_node");
    ros::NodeHandle n2;
    ros::Subscriber sub = n2.subscribe("predictions", 1000, predictionsCallback);       
    ros::Subscriber sub2 = n2.subscribe("/gslicr/segmentation", 1000, labelsCallback);       
    image_transport::ImageTransport it_gslicr(n2);
        ros::Time t;

    image_transport::Publisher pub_avg = it_gslicr.advertise("/final_image", 1);

    // get mask from callback function and averaged image (27X27)
    ros::Rate loop_rate(30);
    while(ros::ok())
    {
        ros::spinOnce();
 
    //splfit(mask); // actual function , use only if needed

    int red_sum[676]={0},green_sum[676]={0},blue_sum[676]={0};
    int n=26;
    int k=0;
    for(int i=0;i<676;i++)
    {
        if(i/n==0 || i%n==0 || i/n==n-1 || i%n==n-1){
            red_sum[i] =0;
            green_sum[i] =0;
            blue_sum[i] =0;
        }
        else {
            red_sum[i] =mask[k]*255;
            green_sum[i] =mask[k]*255;
            blue_sum[i] =mask[k]*255;
            k++;
        }

    }
    
    cv::Mat M1(500,500, CV_8UC3, {0,0,0});
    for (int x=0;x<500;x++)
    {
        for(int y=0;y<500;y++)
        {
                M1.at<cv::Vec3b>(x,y)[0]=blue_sum[labels[x*500 + y ]] ;// b
                M1.at<cv::Vec3b>(x,y)[1]=green_sum[labels[x*500 + y ]] ;// g
                M1.at<cv::Vec3b>(x,y)[2]=red_sum[labels[x*500 + y ]] ;// r
        }
    }
    //cv::namedWindow("video",1);
    //cv::imgshow("video",M1);
    string frame_id="camera";
    t = ros::Time::now();


    pub_avg.publish(imageToROSmsg(M1, sensor_msgs::image_encodings::BGR8, frame_id, t));
    loop_rate.sleep();

	}
 return 0;
}
