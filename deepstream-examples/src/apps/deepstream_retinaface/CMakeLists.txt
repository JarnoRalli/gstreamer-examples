find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Deepstream REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(GST REQUIRED gstreamer-1.0>=1.4)

add_executable(deepstream_retinaface main.cpp)

target_include_directories(deepstream_retinaface
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

target_link_libraries(deepstream_retinaface
        Deepstream::nvds_tools
        TensorRT::TensorRT
        Deepstream::Deepstream
        Deepstream::tools
        CUDA::cudart
        nvtx3
        ${GST_LIBRARIES})

set_target_properties(deepstream_retinaface
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps")

