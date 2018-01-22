 #include "ros/ros.h"
 #include "classifier/lane_classifier.h"
 #include <cstdlib>
 #include <iostream>
 #include <fstream>
 #include <time.h> 

 using namespace :: std;
 int main(int argc, char **argv)
 {
 	  ros::init(argc, argv, "classifier_client");
 	  if (argc != 3)
 	  {
 	    ROS_INFO("usage: classifier_client");
 	    return 1;
 	  }

 	  ros::NodeHandle n;
 	  ros::ServiceClient client = n.serviceClient<classifier::lane_classifier>("classifier");
 	  classifier::lane_classifier srv;
 	  srv.request.data.clear();
 	// ==========================================================


 	  clock_t t;
    t = clock();
    int mask[729];
 	
  //==========================================================
 	
  	for(int i=0;i<27 * 729;i++)
 			srv.request.data.push_back(0); // load all data 
 		
 	
  //===========================================================

   	if (client.call(srv))
   	{
   		for(int i=0;i<729;i++)
   	  	mask[i]=(int)srv.response.ans[i]; // get whole mask
   	}
   	
    else
   	{
     	ROS_ERROR("Failed to call service classifier");
     	return 1;
   	}

   	for (int i=0;i<729;i++)cout<<mask[i]<<" "; // prints the mask

   	srv.request.data.clear();

   	cout<<endl;
   	t = clock() - t;
   	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
   	cout<<(1/time_taken)<<" FPS of prediction"<<endl;
 
    return 0;
 }
