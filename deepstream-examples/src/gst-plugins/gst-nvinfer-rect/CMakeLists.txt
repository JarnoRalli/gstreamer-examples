find_package(CUDAToolkit 11 REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Deepstream REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(GST REQUIRED gstreamer-1.0>=1.4)

add_library(nvdsgst_infer_rect SHARED
    gstnvinfer.cpp
    gstnvinfer_allocator.cpp
    gstnvinfer_property_parser.cpp
    gstnvinfer_meta_utils.cpp gstnvinfer_impl.cpp
    gstnvinfer_yaml_parser.cpp
)

target_include_directories(nvdsgst_infer_rect
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GST_INCLUDE_DIRS}
)

target_link_libraries(nvdsgst_infer_rect
        Deepstream::nvds_tools
        TensorRT::TensorRT
        Deepstream::Deepstream
        CUDA::cudart
        CUDA::nvtx3
        ${YAML_CPP_LIBRARIES}
        ${GST_LIBRARIES}
    )

set_target_properties(nvdsgst_infer_rect
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
)
