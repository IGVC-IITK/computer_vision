cmake_minimum_required(VERSION 2.8.3)
project(classifier)


find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport roscpp rospy std_msgs genmsg sensor_msgs)


#add_message_files(FILES Num.msg)



generate_messages(DEPENDENCIES std_msgs)

catkin_package()


include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(classifier_node src/classifier_node.cpp)
target_link_libraries(classifier_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(classifier_node classifier_gencpp)
