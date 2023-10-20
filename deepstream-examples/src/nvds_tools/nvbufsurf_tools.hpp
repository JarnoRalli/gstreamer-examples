#pragma once

class NvBufSurface;

namespace nvdsutils{

/**
 * @brief Writes images to a file. Allocates memory in the cpu/system and copies the
 * the images to the allocated memory, and then writes the images to a file. Works
 * only with 8-bit gray images.
 * @param[in] surf : buffer containing the images to be written to a file.
 * @param[in] filename : base filename, buffer 0 will be written to <filename>0.png etc.
 * @param[in] use_pitch_alignment: if true, uses surf buffers pitch when mapping cv::Mat.
 * If false, uses OpenCV's internal alignment.
 * @return 1 indicates success.
 */

int write_surfgray8_to_disk(NvBufSurface* surf, const char* filename, bool use_pitch_alignment);

}
