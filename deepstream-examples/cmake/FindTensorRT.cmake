# source:
# https://github.com/NVIDIA/tensorrt-laboratory/blob/master/cmake/FindTensorRT.cmake

# This module defines the following variables:
#
# ::
#
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_FOUND
#
# ::
#
#   TensorRT_VERSION_STRING - version (x.y.z)
#   TensorRT_VERSION_MAJOR  - major version (x)
#   TensorRT_VERSION_MINOR  - minor version (y)
#   TensorRT_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``TensorRT_DIR`` to an installation root to tell this module where to look.
#

if(NOT TensorRT_DIR)
    set(TensorRT_DIR "/usr" PATH)
endif()

#------------------------
# Find include directory
#------------------------
find_path(TensorRT_INCLUDE_DIR NAMES NvInferVersion.h HINTS ${TensorRT_DIR} PATH_SUFFIXES include)

#----------------------
# Find library nvinfer
#----------------------
find_library(TensorRT_LIBRARY NAMES nvinfer HINTS ${TensorRT_DIR} PATH_SUFFIXES lib)

#------------------------
# Find library nvpersers
#------------------------
find_library(TensorRT_NVPARSERS_LIBRARY NAMES nvparsers HINTS ${TensorRT_DIR} PATH_SUFFIXES lib)

#---------------------------
# Find library nvonnxparser
#---------------------------
find_library(TensorRT_NVONNXPARSER_LIBRARY NAMES nvonnxparser HINTS ${TensorRT_DIR} PATH_SUFFIXES lib)

#-----------------------------
# Find library nvinfer_plugin
#-----------------------------
find_library(TensorRT_PLUGIN_LIBRARY NAMES nvinfer_plugin HINTS ${TensorRT_DIR} PATH_SUFFIXES lib)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}" CACHE STRING "TensorRT version" FORCE)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIR
        TensorRT_LIBRARY
        TensorRT_NVPARSERS_LIBRARY
        TensorRT_NVONNXPARSER_LIBRARY
        TensorRT_PLUGIN_LIBRARY
        VERSION_VAR
        TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

    if(NOT TensorRT_LIBRARIES)
        set(TensorRT_LIBRARIES ${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY} ${TensorRT_NVONNXPARSER_LIBRARY} ${TensorRT_NVPARSERS_LIBRARY})
    endif()

    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
        set_property(TARGET TensorRT::TensorRT APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
        target_link_libraries(TensorRT::TensorRT
            INTERFACE
                ${TensorRT_LIBRARY}
                ${TensorRT_PLUGIN_LIBRARY}
                ${TensorRT_NVONNXPARSER_LIBRARY}
                ${TensorRT_NVPARSERS_LIBRARY})
    endif()
endif()
