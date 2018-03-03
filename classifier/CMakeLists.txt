cmake_minimum_required(VERSION 2.8.3)
project(classifier)


find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport roscpp rospy std_msgs genmsg sensor_msgs)
add_service_files(FILES lane_classifier.srv )

#add_message_files(FILES Num.msg)



generate_messages(DEPENDENCIES std_msgs)

catkin_package()


include_directories(include ${catkin_INCLUDE_DIRS})


add_executable(master src/master.cpp)
target_link_libraries(master ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(master classifier_gencpp)