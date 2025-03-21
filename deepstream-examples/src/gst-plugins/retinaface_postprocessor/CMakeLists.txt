find_package(CUDAToolkit REQUIRED)
find_package(Deepstream REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(GST REQUIRED gstreamer-1.0>=1.4)
pkg_search_module(GST_BASE REQUIRED gstreamer-base-1.0>=1.4)
pkg_check_modules(GLIB REQUIRED glib-2.0)

add_library(gstretinaface_postprocessor SHARED
    gstretinafacepostprocessor.cpp
)

target_compile_options(gstretinaface_postprocessor PRIVATE -fPIC -Wall -Wextra)
target_compile_definitions(gstretinaface_postprocessor PRIVATE HAVE_CONFIG_H)

target_include_directories(gstretinaface_postprocessor
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GST_INCLUDE_DIRS}
        ${GST_BASE_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
)

target_link_libraries(gstretinaface_postprocessor
        Deepstream::nvds_tools
        Deepstream::Deepstream
        CUDA::cudart
        ${YAML_CPP_LIBRARIES}
        ${GST_LIBRARIES}
        ${GST_BASE_LIBRARIES}
        ${GLIB_LIBRARIES}
)

set_target_properties(gstretinaface_postprocessor
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gst_plugins"
)
