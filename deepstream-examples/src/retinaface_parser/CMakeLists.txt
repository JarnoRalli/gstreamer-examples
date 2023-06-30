find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Deepstream REQUIRED)

add_library(nvdsparser_retinaface SHARED nvdsparse_retinaface.cpp)

target_link_libraries(nvdsparser_retinaface TensorRT::TensorRT Deepstream::Deepstream CUDA::cudart)
