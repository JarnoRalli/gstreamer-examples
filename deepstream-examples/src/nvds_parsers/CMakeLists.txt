find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Deepstream REQUIRED)

add_library(nvdsparser_retinaface SHARED nvdsparser_retinaface.cpp)

target_link_libraries(nvdsparser_retinaface
    TensorRT::TensorRT
    Deepstream::Deepstream
    Deepstream::tools
    CUDA::cudart)

set_target_properties(nvdsparser_retinaface
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvds_parsers"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvds_parsers"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvds_parsers")

add_library(nvdsparser_dummy SHARED nvdsparser_dummy.cpp)

target_link_libraries(nvdsparser_dummy
    TensorRT::TensorRT
    Deepstream::Deepstream
    Deepstream::tools
    CUDA::cudart)

set_target_properties(nvdsparser_dummy
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvds_parsers"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvds_parsers"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvds_parsers")

