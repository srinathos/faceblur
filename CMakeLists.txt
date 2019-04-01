#CMAKE MINIMUM CHECK
cmake_minimum_required(VERSION 3.8)

# defining project name
project(FaceBlur)

# Setting parameters
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES faceblur.cpp)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0/)
set(DLIB_USE_CUDA 1)
add_executable(FaceBlur ${SOURCE_FILES})
# Adding dependencies
add_subdirectory(../lib/dlib/dlib dlib_build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Linking dependencies to source code.
target_link_libraries(FaceBlur dlib::dlib)
target_link_libraries( FaceBlur ${OpenCV_LIBS} )