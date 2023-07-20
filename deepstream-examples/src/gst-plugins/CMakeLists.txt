find_package(Deepstream REQUIRED)

if(NOT DEFINED Deepstream_VERSION_MAJOR)
    message(FATAL_ERROR "Variable 'Deepstream_VERSION_MAJOR' does not exist")
endif()

if(NOT DEFINED Deepstream_VERSION_MINOR)
    message(FATAL_ERROR "Variable 'Deepstream_VERSION_MINOR' does not exist")
endif()

set(Deepstream_directory "deepstream${Deepstream_VERSION_MAJOR}.${Deepstream_VERSION_MINOR}")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${Deepstream_directory})
    add_subdirectory(${Deepstream_directory})
else()
    message(FATAL_ERROR "Directory '${CMAKE_CURRENT_SOURCE_DIR}/${Deepstream_directory}' does not exist")
endif()
