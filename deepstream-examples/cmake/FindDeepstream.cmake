# This module defines the following variables:
#
# ::
#
#   Deepstream_INCLUDE_DIRS
#   Deepstream_LIBRARIES
#   Deepstream_FOUND
#
# ::
#
#   Deepstream_VERSION_STRING - version (x.y.z)
#   Deepstream_VERSION_MAJOR  - major version (x)
#   Deepstream_VERSION_MINOR  - minor version (y)
#   Deepstream_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``Deepstream_DIR`` to an installation root to tell this module where to look.
#
set(_Deepstream_SEARCHES)

if(Deepstream_DIR)
    set(_Deepstream_SEARCH_ROOT PATHS ${Deepstream_DIR} NO_DEFAULT_PATH)
    list(APPEND _Deepstream_SEARCHES _Deepstream_SEARCH_ROOT)
endif()

# appends some common paths
set(_Deepstream_SEARCH_NORMAL
        PATHS "/opt/nvidia/deepstream/deepstream"
        )
list(APPEND _Deepstream_SEARCHES _Deepstream_SEARCH_NORMAL)

# Include dir
foreach(search ${_Deepstream_SEARCHES})
    find_path(Deepstream_INCLUDE_DIR NAMES nvdsinfer.h nvdsinfer_custom_impl.h ${${search}} PATH_SUFFIXES sources/includes)
endforeach()

if(NOT Deepstream_LIBRARY)
    foreach(search ${_Deepstream_SEARCHES})
        find_library(Deepstream_LIBRARY NAMES nvds_infer ${${search}} PATH_SUFFIXES lib)
    endforeach()
endif()

if(NOT Deepstream_PARSERS_LIBRARY)
    foreach(search ${_Deepstream_SEARCHES})
        find_library(Deepstream_NVPARSERS_LIBRARY NAMES nvparsers ${${search}} PATH_SUFFIXES lib)
    endforeach()
endif()

if(NOT Deepstream_NVONNXPARSER_LIBRARY)
    foreach(search ${_Deepstream_SEARCHES})
        find_library(Deepstream_NVONNXPARSER_LIBRARY NAMES nvonnxparser ${${search}} PATH_SUFFIXES lib)
    endforeach()
endif()

if(NOT Deepstream_PLUGIN_LIBRARY)
    foreach(search ${_Deepstream_SEARCHES})
        find_library(Deepstream_PLUGIN_LIBRARY NAMES nvinfer_plugin ${${search}} PATH_SUFFIXES lib)
    endforeach()
endif()

mark_as_advanced(Deepstream_INCLUDE_DIR)

if(Deepstream_INCLUDE_DIR AND EXISTS "${Deepstream_INCLUDE_DIR}/nvds_version.h")
    file(STRINGS "${Deepstream_INCLUDE_DIR}/nvds_version.h" Deepstream_MAJOR REGEX "^#define NVDS_VERSION_MAJOR [0-9]+.*$")
    file(STRINGS "${Deepstream_INCLUDE_DIR}/nvds_version.h" Deepstream_MINOR REGEX "^#define NVDS_VERSION_MINOR [0-9]+.*$")
    file(STRINGS "${Deepstream_INCLUDE_DIR}/nvds_version.h" Deepstream_MICRO REGEX "^#define NVDS_VERSION_MICRO [0-9]+.*$")

    string(REGEX REPLACE "^#define NVDS_VERSION_MAJOR ([0-9]+).*$" "\\1" Deepstream_VERSION_MAJOR "${Deepstream_MAJOR}")
    string(REGEX REPLACE "^#define NVDS_VERSION_MINOR ([0-9]+).*$" "\\1" Deepstream_VERSION_MINOR "${Deepstream_MINOR}")
    string(REGEX REPLACE "^#define NVDS_VERSION_MICRO ([0-9]+).*$" "\\1" Deepstream_VERSION_MICRO "${Deepstream_MICRO}")
    set(Deepstream_VERSION_STRING "${Deepstream_VERSION_MAJOR}.${Deepstream_VERSION_MINOR}.${Deepstream_VERSION_MICRO}" CACHE STRING "Deepstream version" FORCE)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Deepstream REQUIRED_VARS Deepstream_LIBRARY Deepstream_INCLUDE_DIR VERSION_VAR Deepstream_VERSION_STRING)

if(Deepstream_FOUND)
    set(Deepstream_INCLUDE_DIRS ${Deepstream_INCLUDE_DIR})

    if(NOT Deepstream_LIBRARIES)
        set(Deepstream_LIBRARIES ${Deepstream_LIBRARY} ${Deepstream_PLUGIN_LIBRARY} ${Deepstream_NVONNXPARSER_LIBRARY} ${Deepstream_NVPARSERS_LIBRARY})
    endif()

    if(NOT TARGET Deepstream::Deepstream)
        add_library(Deepstream::Deepstream UNKNOWN IMPORTED)
        set_target_properties(Deepstream::Deepstream PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Deepstream_INCLUDE_DIRS}")
        set_property(TARGET Deepstream::Deepstream APPEND PROPERTY IMPORTED_LOCATION "${Deepstream_LIBRARY}")
    endif()
endif()
