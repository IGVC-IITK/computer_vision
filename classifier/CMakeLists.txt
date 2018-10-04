cmake_minimum_required(VERSION 2.8.3)
project(classifier)


find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport roscpp rospy std_msgs genmsg sensor_msgs OpenCV)
add_service_files(FILES lane_classifier.srv )

#add_message_files(FILES Num.msg)



generate_messages(DEPENDENCIES std_msgs)

catkin_package()


include_directories(include ${catkin_INCLUDE_DIRS})


add_executable(master src/master.cpp)
target_link_libraries(master ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(master classifier_gencpp)

add_executable(new_master src/new_master.cpp)
target_link_libraries(new_master ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(new_master classifier_gencpp)

add_executable(thresholder src/thresholder.cpp)
target_link_libraries(thresholder ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(thresholder classifier_gencpp)