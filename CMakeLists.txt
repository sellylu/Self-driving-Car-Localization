cmake_minimum_required(VERSION 2.8.3)
project(localization)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf
  tf_conversions
)
find_package(PCL 1.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()

## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES localization
#  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs tf tf_conversions
#  DEPENDS system_lib
  DEPENDS 
  Eigen
  OpenMP
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(${PROJECT_NAME}_node src/localization_node.cpp)

target_compile_options(${PROJECT_NAME}_node PRIVATE ${OpenMP_FLAGS})
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${OpenMP_LIBS}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_hw5_a071715.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()


## Add folders to be run by python nosetests
# catkin_add_nosetests(test)