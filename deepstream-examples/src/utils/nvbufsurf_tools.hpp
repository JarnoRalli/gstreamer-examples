#pragma once

class NvBufSurface;

namespace nvdsutils{

/**
 * @brief Writes the images in the surf-buffer to PNG files.
 * @param[in] surf : buffer containing the images to be written to a file.
 * @param[in] filename : base filename, buffer 0 will be written to <filename>0.png etc.
 * @return 1 indicates success.
 * @todo this function only works if the surface is already in RGB (planar)
 * color-format. In order for this to be more generic, the surface
 * should be converted into a suitable format if it's not. Colours
 * will be mangled, since OpenCV expects BGR images.
 */
int write_nvbufsurface_to_png(NvBufSurface* surf, const char* filename);

}
