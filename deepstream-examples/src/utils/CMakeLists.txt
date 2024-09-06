find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Deepstream REQUIRED)

add_library(nvds_tools SHARED nvbufsurf_tools.hpp nvbufsurf_tools.cpp)

target_include_directories(nvds_tools
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(nvds_tools
    PUBLIC
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    TensorRT::TensorRT
    Deepstream::Deepstream
    CUDA::nppig
    )

add_library(Deepstream::nvds_tools ALIAS nvds_tools)

