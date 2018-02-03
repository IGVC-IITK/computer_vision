cmake_minimum_required(VERSION 2.8.3)
project(classifier)


find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport roscpp rospy std_msgs genmsg sensor_msgs)


#add_message_files(FILES Num.msg)



generate_messages(DEPENDENCIES std_msgs)

catkin_package()


include_directories(include ${catkin_INCLUDE_DIRS})

##add_executable(talker src/talker.cpp)
##target_link_libraries(talker ${catkin_LIBRARIES})

##add_executable(listener src/listener.cpp)
##target_link_libraries(listener ${catkin_LIBRARIES})

#add_executable(client src/client.cpp)
#target_link_libraries(client ${catkin_LIBRARIES})
#add_dependencies(client classifier_gencpp)

add_executable(master src/master.cpp)
target_link_libraries(master ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(master classifier_gencpp)
