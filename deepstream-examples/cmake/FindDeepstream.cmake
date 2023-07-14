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

if(NOT Deepstream_DIR)
    set(Deepstream_DIR "/opt/nvidia/deepstream/deepstream" PATH)
endif()

#------------------------
# Find include directory
#------------------------
find_path(Deepstream_INCLUDE_DIR NAMES nvds_version.h HINTS ${Deepstream_DIR} PATH_SUFFIXES sources/includes)

#----------------------------------
# Find nvdsinfer include directory
#----------------------------------
find_path(Deepstream_INCLUDE_DIR_INFER_UTILS NAMES nvdsinfer_func_utils.h HINTS ${Deepstream_DIR} PATH_SUFFIXES sources/libs/nvdsinfer)

#----------------------
# Find nvinfer library
#----------------------
find_library(Deepstream_INFER_LIBRARY NAMES nvds_infer HINTS ${Deepstream_DIR} PATH_SUFFIXES lib)

#----------------------
# Find nvdsgst_helper
#----------------------
find_library(Deepstream_GST_HELPER_LIBRARY NAMES nvdsgst_helper HINTS ${Deepstream_DIR} PATH_SUFFIXES lib)

#-------------------
# Find nvdsgst_meta
#-------------------
find_library(Deepstream_GST_META_LIBRARY NAMES nvdsgst_meta HINTS ${Deepstream_DIR} PATH_SUFFIXES lib)

#-------------------
# Find nvds_meta
#-------------------
find_library(Deepstream_META_LIBRARY NAMES nvds_meta HINTS ${Deepstream_DIR} PATH_SUFFIXES lib)

#----------------------
# Find nvdsbufsurf
#----------------------
find_library(Deepstream_BUFSURFACE_LIBRARY NAMES nvbufsurface HINTS ${Deepstream_DIR} PATH_SUFFIXES lib)

#---------------------------
# Find nvdsbufsurftransform
#---------------------------
find_library(Deepstream_BUFSURFTRANSFORM_LIBRARY NAMES nvbufsurftransform HINTS ${Deepstream_DIR} PATH_SUFFIXES lib)

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
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Deepstream
    REQUIRED_VARS
        Deepstream_INCLUDE_DIR
        Deepstream_INCLUDE_DIR_INFER_UTILS
        Deepstream_INFER_LIBRARY
        Deepstream_GST_HELPER_LIBRARY
        Deepstream_GST_META_LIBRARY
        Deepstream_META_LIBRARY
        Deepstream_BUFSURFACE_LIBRARY
        Deepstream_BUFSURFTRANSFORM_LIBRARY
        VERSION_VAR
        Deepstream_VERSION_STRING)

if(Deepstream_FOUND)
    set(Deepstream_INCLUDE_DIRS "${Deepstream_INCLUDE_DIR};${Deepstream_INCLUDE_DIR_INFER_UTILS}" CACHE STRING "Deepstream INCLUDE directories" FORCE)

    if(NOT Deepstream_LIBRARIES)
        set(Deepstream_LIBRARIES ${Deepstream_INFER_LIBRARY})
    endif()

    if(NOT TARGET Deepstream::Deepstream)
        add_library(Deepstream::Deepstream UNKNOWN IMPORTED)
        set_target_properties(Deepstream::Deepstream PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Deepstream_INCLUDE_DIRS}")
        set_property(TARGET Deepstream::Deepstream APPEND PROPERTY IMPORTED_LOCATION "${Deepstream_INFER_LIBRARY}")
        target_link_libraries(Deepstream::Deepstream
            INTERFACE
                ${Deepstream_INFER_LIBRARY}
                ${Deepstream_GST_HELPER_LIBRARY}
                ${Deepstream_GST_META_LIBRARY}
                ${Deepstream_META_LIBRARY}
                ${Deepstream_BUFSURFACE_LIBRARY}
                ${Deepstream_BUFSURFTRANSFORM_LIBRARY}
        )
    endif()
endif()
