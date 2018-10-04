cmake_minimum_required(VERSION 2.8.3)
project(potholes)

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

add_executable(detect_potholes src/detect_potholes.cpp)
target_link_libraries(detect_potholes ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(detect_potholes potholes_gencpp)
