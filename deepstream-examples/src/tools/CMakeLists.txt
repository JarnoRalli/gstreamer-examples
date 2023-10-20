add_library(tools SHARED
    data_types.hpp
    data_types.cpp
    nms.hpp
    nms.cpp)

target_include_directories(tools
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

add_library(Deepstream::tools ALIAS tools)

