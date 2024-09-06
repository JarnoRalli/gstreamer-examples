#include <string>
#include <cstdint>
#include <iostream>
#include <exception>

#include "nvbufsurf_tools.hpp"
#include "nvbufsurface.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace nvdsutils{

int write_surfgray8_to_disk(NvBufSurface* surf, const char* filename, bool use_pitch_alignment)
{
    NvBufSurface* drawMe = nullptr;

    if((surf == nullptr) || (surf->numFilled <=0))
    {
        std::cerr << "write_surfgray8_to_disk: surf is nullptr or numFilled is 0" << std::endl;
        return -1;
    }

    if(surf->surfaceList[0].colorFormat != NVBUF_COLOR_FORMAT_GRAY8)
    {
        std::cerr << "write_surfgray8_to_disk: only NVBUF_COLOR_FORMAT_GRAY8 is supported, current format is " << surf->surfaceList[0].colorFormat << std::endl;
        return -1;
    }

    // Allocate system memory and copy the surface buffers to the allocated memory
    NvBufSurfaceCreateParams surf_create_params = {
        .gpuId = surf->gpuId,
        .width = surf->surfaceList[0].width,
        .height = surf->surfaceList[0].height,
        .size = 0,
        .isContiguous = true,
        .colorFormat = NVBUF_COLOR_FORMAT_GRAY8,
        .layout = NVBUF_LAYOUT_PITCH,
        .memType = NVBUF_MEM_SYSTEM
    };

    // Allocate memory
    if(NvBufSurfaceCreate(&drawMe, surf->batchSize, &surf_create_params) != 0)
    {
        std::cerr << "write_surfgray8_to_disk: failed to allocate memory" << std::endl;
        return -1;
    }

    // Copy the buffers to the allocated memory
    if(NvBufSurfaceCopy(surf, drawMe) != 0)
    {
        std::cerr << "write_surfgray8_to_disk: failed to copy buffers" << std::endl;
        NvBufSurfaceDestroy(drawMe);
        return -1;
    }

    // Map the images to OpenCV cv::Mat and write the buffer contents to file(s)
    std::string fileName(filename);
    for(std::size_t i = 0; i < surf->numFilled; i++)
    {
        std::string bufferFileName = fileName + "_object_" + std::to_string(i) + ".bmp";
        cv::Mat mapped;

        if(use_pitch_alignment)
        {   mapped = cv::Mat(drawMe->surfaceList[i].height, drawMe->surfaceList[i].width, CV_8UC1, drawMe->surfaceList[i].dataPtr, drawMe->surfaceList[i].pitch);
        }else{
            mapped = cv::Mat(drawMe->surfaceList[i].height, drawMe->surfaceList[i].width, CV_8UC1, drawMe->surfaceList[i].dataPtr);
        }

        try{
            cv::imwrite(bufferFileName, mapped);
        }catch(const std::exception& e)
        {
            NvBufSurfaceDestroy(drawMe);
            std::cerr << "write_surfgray8_to_disk exception thrown during saving the image: " << e.what() << std::endl;
            return -1;
        }
    }

    // Release memory
    NvBufSurfaceDestroy(drawMe);

    return 1;
}

}
