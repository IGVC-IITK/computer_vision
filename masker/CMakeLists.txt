cmake_minimum_required(VERSION 2.8.3)
project(masker)

find_package(catkin REQUIRED 
OpenCV REQUIRED
COMPONENTS
cv_bridge image_transport
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

add_executable(masker_node src/masker_node.cpp)
target_link_libraries(masker_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(masker_node masker_gencpp)
