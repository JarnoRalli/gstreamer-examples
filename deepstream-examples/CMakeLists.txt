cmake_minimum_required(VERSION 3.0)

project(Deepstream_extensions VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 CACHE STRING "CUDA architectures" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(BUILD_RETINAFACE_PARSER "Build RetinaFace perser" ON)

if(BUILD_RETINAFACE_PARSER)
    add_subdirectory(src/retinaface_parser)
endif()
