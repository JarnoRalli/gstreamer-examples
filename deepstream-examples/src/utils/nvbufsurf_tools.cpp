#include <string>
#include <cstdint>
#include <iostream>

#include "nvbufsurf_tools.hpp"
#include "nvbufsurface.h"
#include "opencv2/imgcodecs.hpp"

namespace nvdsutils{

//TODO: this function only works if the surface is already in RGB (planar)
// color-format. In order for this to be more generic, the surface
// should be converted into a suitable format if it's not.
int write_nvbufsurface_to_png(NvBufSurface* surf, const char* filename)
{
    if((surf == nullptr) || (surf->numFilled <=0))
    {
        std::cerr << "write_nvbufsurface_to_png: surf is nullptr or numFilled is 0" << std::endl;
        return -1;
    }

    NvBufSurface* host_temp;
    NvBufSurfaceCreateParams host_temp_params;
    int width = surf->surfaceList[0].width;
    int height = surf->surfaceList[0].height;
    int len = width*height*3;

    host_temp_params.gpuId = surf->gpuId;
    host_temp_params.width = width;
    host_temp_params.height = height;
    host_temp_params.size = 0;
    host_temp_params.isContiguous = true;
    host_temp_params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
    host_temp_params.memType = NVBUF_MEM_SYSTEM;
    host_temp_params.layout = NVBUF_LAYOUT_PITCH;

    if(NvBufSurfaceCreate(&host_temp, surf->batchSize, &host_temp_params) != 0)
    {
        std::cerr << "write_nvbufsurface_to_png: failed to allocate memory" << std::endl;
        return -1;
    }

    if(NvBufSurfaceCopy(surf, host_temp) != 0)
    {
        std::cerr << "write_nvbufsurface_to_png: failed to copy buffers" << std::endl;
        return -1;
    }

    std::string fileName(filename);
    for(std::size_t i = 0; i <= surf->numFilled; i++)
    {
        std::string bufferFileName = fileName + std::to_string(i) + ".png";
        std::cout << bufferFileName << std::endl;
        //TODO: OpenCV expects the image to be in BRG format, so the colours will be wrong
        cv::Mat mapped(height, width, CV_8UC3, host_temp->surfaceList[0].dataPtr, host_temp->surfaceList[0].pitch);
        cv::imwrite(bufferFileName, mapped);
    }

    NvBufSurfaceDestroy(host_temp);

    return 1;
}

}
