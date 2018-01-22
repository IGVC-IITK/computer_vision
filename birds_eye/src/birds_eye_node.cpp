#include <fstream>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

using namespace std;

void imageCallback(const sensor_msgs::ImageConstPtr& imgMessage, cv::Mat& image)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(imgMessage, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  image = cv_ptr->image;
}


int main(int argc, char* argv[])
{
  ofstream out("~/.ros/top_view.txt");

  ros::init(argc, argv, "birds_eye_node");

  ros::NodeHandle nh;
  std::string image_topic;
  if (argc < 2)
  {
      cerr << endl << "Usage: rosrun birds_eye birds_eye_node [image_topic:=/input/topic] board_w board_h" << endl;        
      ros::shutdown();
      return 1;
  }

  cv::Mat image;
  image_transport::ImageTransport it_bird(nh);
  image_transport::Subscriber sub_img = it_bird.subscribe("/image_topic", 1, boost::bind(imageCallback, _1, boost::ref(image)));

  int board_w = atoi(argv[1]);
  int board_h = atoi(argv[2]);

  cv::Size board_sz(board_w, board_h);
  cv::Mat gray_image, tmp, H , birds_image, birds_blank;
  cv::Point2f objPts[4], imgPts[4], center;
  std::vector<cv::Point2f> corners;
  image.create(cv::Size(640, 480), CV_8UC3);
  float Z = 1; //have experimented from values as low as .1 and as high as 100
  int key = 0;

  ros::Rate loop_rate(1);
  while (ros::ok())
  {
    ros::spinOnce();
    cv::Mat blank = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
    int found = cv::findChessboardCorners(image, board_sz, corners);

    if (found)
    {
      cv::drawChessboardCorners(image, board_sz, corners, 1);
      cv::cvtColor(image, gray_image, CV_RGB2GRAY);
      cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      cv::resize(image, tmp, cv::Size(), .5, .5);
      cv::namedWindow("IMAGE");
      cv::imshow("IMAGE" , tmp);
      break;
    }
    else
      cout<<"No board detected"<<endl;

    loop_rate.sleep();
  }
  cv::waitKey(0);

  imgPts[0] = corners.at(0); 
  imgPts[1] = corners.at(board_w-1);
  imgPts[2] = corners.at((board_h-1) * board_w);
  imgPts[3] = corners.at((board_h-1) * board_w + board_w-1);
  center.x= (imgPts[0].x+imgPts[1].x+imgPts[2].x+imgPts[3].x)/4;
  center.y= (imgPts[0].y+imgPts[1].y+imgPts[2].y+imgPts[3].y)/4;


  int q1,q2,q3,q4;
  for (int i = 0; i < 4; ++i)
  {
    if ((imgPts[i].x>center.x)&&(imgPts[i].y>center.y))
    {
      q1=i;
    }
    else if ((imgPts[i].x<center.x)&&(imgPts[i].y>center.y))
    {
      q2=i;
    }
    else if ((imgPts[i].x<center.x)&&(imgPts[i].y<center.y))
    {
      q3=i;
    }
    else if ((imgPts[i].x>center.x)&&(imgPts[i].y<center.y))
    {
      q4=i;
    }
  }
  
  objPts[q3].x = 420;
  objPts[q3].y = 540;
  objPts[q4].x = 860;
  objPts[q4].y = 540;
  objPts[q2].x = 420;
  objPts[q2].y = 1244;
  objPts[q1].x = 860;
  objPts[q1].y = 1244;

  H = cv::getPerspectiveTransform( objPts, imgPts );
  out<<H<<endl;
  cout<<H<<endl;
  out.close();
  birds_image = image;
  cout<<"mark"<<endl;
  
  while (key != 27)
  {
    cout<<"mark"<<endl;
    H.at<float>(2,2) = Z;

    cv::warpPerspective(image, birds_image, H, cv::Size( 640,480 ) ,
                         CV_INTER_LINEAR | CV_WARP_INVERSE_MAP | CV_WARP_FILL_OUTLIERS);
    
    cout<<"mark"<<endl;
    for (int i = 0; i < 6; ++i)
    {
      cout<<i<<"    "<<corners.at(i*board_w)<<"     "<<corners.at((i+1)*board_w-1)<<endl;
    }
    cv::imshow("IMAGE", birds_image);
    cv::waitKey(0);
    break;  
  }

  return 0;
}
