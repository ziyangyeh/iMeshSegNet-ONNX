cmake_minimum_required(VERSION 3.18)

project(iMeshSegNet-ONNX LANGUAGES C CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(Open3D_DIR open3d/lib/cmake/Open3D/)
set(Torch_DIR libtorch/share/cmake/Torch/)

find_package(Torch REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

find_library(NVINFER  NAMES nvinfer)
find_library(NVINFERPLUGIN NAMES nvinfer_plugin)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSER NAMES nvonnxparser)

include_directories("${CMAKE_SOURCE_DIR}/include/")
file(GLOB SRC_DIR "${CMAKE_SOURCE_DIR}/src/*.cpp") 

include_directories("${CMAKE_SOURCE_DIR}/gco_src/")
file(GLOB GCO_DIR "${CMAKE_SOURCE_DIR}/gco_src/GCoptimization.cpp" "${CMAKE_SOURCE_DIR}/gco_src/LinkedBlockList.cpp")

include_directories(
  include
  ${Open3D_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(TeethLabeler SHARED ${SRC_DIR} ${GCO_DIR})
target_link_libraries(TeethLabeler "${TORCH_LIBRARIES}" Open3D::Open3D OpenMP::OpenMP_CXX nvinfer nvinfer_plugin nvparsers nvonnxparser)

cuda_add_executable(Test main.cpp ${SRC_DIR} ${GCO_DIR})
target_link_libraries(Test "${TORCH_LIBRARIES}" Open3D::Open3D OpenMP::OpenMP_CXX nvinfer nvinfer_plugin nvparsers nvonnxparser)
