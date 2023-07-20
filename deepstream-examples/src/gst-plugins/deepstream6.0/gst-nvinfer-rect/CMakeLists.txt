find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Deepstream REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(GST REQUIRED 
    gstreamer-1.0>=1.4
    gstreamer-base-1.0>=1.4
    gstreamer-video-1.0>=1.4)

add_library(nvdsgst_inferrect SHARED
    gstnvinfer.cpp
    gstnvinfer_allocator.cpp
    gstnvinfer_property_parser.cpp
    gstnvinfer_meta_utils.cpp
    gstnvinfer_impl.cpp
)

target_include_directories(nvdsgst_inferrect
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GST_INCLUDE_DIRS}
)

# CMake 3.25 and later define CUDA::nvtx3. Older versions define CUDA::nvToolsExt
if(NOT TARGET nvtx3)
    if(TARGET CUDA::nvtx3)
        add_library(nvtx3 ALIAS CUDA::nvtx3)
    elseif(TARGET CUDA::nvToolsExt)
        add_library(nvtx3 ALIAS CUDA::nvToolsExt)
    else()
        message(ERROR_FATAL "Cannot find neither of the following targets: CUDA::nvtx3 or CUDA::nvToolsExt")
    endif()
endif()

target_link_libraries(nvdsgst_inferrect
        Deepstream::nvds_tools
        TensorRT::TensorRT
        Deepstream::Deepstream
        CUDA::cudart
        CUDA::nvToolsExt
        ${YAML_CPP_LIBRARIES}
        ${GST_LIBRARIES}
)

set_target_properties(nvdsgst_inferrect
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
)
