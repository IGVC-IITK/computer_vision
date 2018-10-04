cmake_minimum_required(VERSION 2.8.3)
project(alt_classifier)

find_package(catkin REQUIRED 
  OpenCV REQUIRED
  COMPONENTS
  cv_bridge 
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

 generate_messages(
   DEPENDENCIES
   std_msgs#   std_msgs
 )

catkin_package(
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

add_executable(thresh_node src/thresh_node.cpp)
target_link_libraries(thresh_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(thresh_node thresh_gencpp)


add_executable(threshnode2 src/threshnode2.cpp)
target_link_libraries(threshnode2 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(threshnode2 thresh_gencpp)