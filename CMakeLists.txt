PROJECT(post-ba)
cmake_minimum_required(VERSION 3.5)

IF(NOT CMAKE_BUILD_TYEP)
 SET(CMAKE_BUILD_TYPE Release)
ENDIF()

SET(CMAKE_CXX_FLAGS_RELEASE "-O3-DNDEBUG")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-pthread")

set(CMAKE_RUNTIME_OUTPUT_DWIRECTORY ${post-ba_SOURCE_DIR}/bin)
SET(SOURCE src/main_linear_xyz.cpp
/home/kh/ORB_SLAM2/src/alglib/alglibinternal.cpp
/home/kh/ORB_SLAM2/src/alglib/diffequations.cpp
/home/kh/ORB_SLAM2/src/alglib/linalg.cpp
/home/kh/ORB_SLAM2/src/alglib/statistics.cpp
/home/kh/ORB_SLAM2/src/alglib/alglibmisc.cpp 
/home/kh/ORB_SLAM2/src/alglib/fasttransforms.cpp
/home/kh/ORB_SLAM2/src/alglib/optimization.cpp
/home/kh/ORB_SLAM2/src/alglib/ap.cpp
/home/kh/ORB_SLAM2/src/alglib/integration.cpp
/home/kh/ORB_SLAM2/src/alglib/solvers.cpp
/home/kh/ORB_SLAM2/src/alglib/specialfunctions.cpp
/home/kh/ORB_SLAM2/src/alglib/interpolation.cpp
/home/kh/ORB_SLAM2/src/alglib/dataanalysis.cpp)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED)
find_package(CSparse)
include_directories(${G2O_INCLUDE_DIRS} 
${CSPARSE_INCLUDE_DIR})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS}
${post-ba_SOURCE_DIR}/include
/home/kh/ORB_SLAM2/src/alglib)

include_directories("/usr/include/eigen3")

add_executable(exe ${SOURCES})
link_directories(${OpencCV_LIBRARY_DIRS})

target_link_libraries(exe ${OpenCV_LIBS}
${G2O_LIB}
${CSPARSE_LIB}
${Sophus_LIBRARIES}
g2o_core
g2o_stuff
g2o_types_sba
g2o_csparse_extension
cxsparse
)




