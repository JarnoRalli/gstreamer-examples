cmake_minimum_required(VERSION 3.17)

project(Deepstream_extensions VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 CACHE STRING "CUDA architectures" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(BUILD_NVDS_PARSERS "Build nvds parsers" ON)
option(BUILD_GST_PLUGINS "Build gst-plugins" ON)
option(BUILD_NVDS_TOOLS "Build deepstream tools" ON)
option(BUILD_TOOLS "Build tools" ON)
option(BUILD_DEEPSTREAM_RETINAFACE_APP "Build Deepstream RetinaFace app" ON)

if(BUILD_NVDS_PARSERS)
    add_subdirectory(src/nvds_parsers)
endif()

if(BUILD_GST_PLUGINS)
    add_subdirectory(src/gst-plugins)
endif()

if(BUILD_NVDS_TOOLS)
    add_subdirectory(src/nvds_tools)
endif()

if(BUILD_TOOLS)
    add_subdirectory(src/tools)
endif()

if(BUILD_DEEPSTREAM_RETINAFACE_APP)
    add_subdirectory(src/apps/deepstream_retinaface)
endif()

