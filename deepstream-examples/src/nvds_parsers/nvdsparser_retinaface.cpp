/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// TODO:
// 1. Verify that works with batch sizes > 1
// 2. Proper error handling


#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include "nvdsinfer_custom_impl.h"
#include "nms.hpp"

// It appears that only a very limited number of parameters can be passed to this function
constexpr float NMS_IOU_THRESHOLD = 0.2f;
// Percentage of padding added around the bounding boxes
constexpr float PADDING_FACTOR = 0.1f;
// Number of landmark points
constexpr unsigned int NR_LANDMARKS = 5;
// Draw landmarks
// You need to modify the nvinfer configuration so that the expected number of classes is 2 in order for this to: num-detected-classes=2
constexpr unsigned int DRAW_LANDMARKS = 0;

// Landmark with 5 keypoints
using Landmark5 = Landmark<NR_LANDMARKS>;

extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

static bool NvDsInferParseRetinaface(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    // Look for the bboxes layer
    auto itr_bbox = std::find_if(outputLayersInfo.begin(), outputLayersInfo.end(), [](const NvDsInferLayerInfo& obj){ return std::string(obj.layerName) == "bboxes";});
    if(itr_bbox == outputLayersInfo.end())
    {
        std::cerr << "Could not find an output layer called 'bboxes'" << std::endl;
        return false;
    }
    
    // Look for the classes layer
    auto itr_class = std::find_if(outputLayersInfo.begin(), outputLayersInfo.end(), [](const NvDsInferLayerInfo& obj){ return std::string(obj.layerName) == "classes";});
    if(itr_class == outputLayersInfo.end())
    {
        std::cerr << "Could not find an output layer called 'classes'" << std::endl;
        return false;
    }
    
    // Look for the landmarks layer
    auto itr_landmark = std::find_if(outputLayersInfo.begin(), outputLayersInfo.end(), [](const NvDsInferLayerInfo& obj){ return std::string(obj.layerName) == "landmarks";});
    if(itr_landmark == outputLayersInfo.end())
    {
        std::cerr << "Could not find an output layer called 'landmarks'" << std::endl;
        return false;
    }

    // Pointers to the buffers
    float* bbox_data = (float *)itr_bbox->buffer;
    float* class_data = (float *)itr_class->buffer;
    float* landmark_data = (float *)itr_landmark->buffer;

    // Verify that bbox:es and landmarks have same number of elements
    if(itr_bbox->inferDims.d[0] != itr_landmark->inferDims.d[0])
    {
        std::cerr << "Nr bbox elements (" << itr_bbox->inferDims.d[0] << ") and nr landmark elements (" << itr_landmark->inferDims.d[0] << ") need to match!" << std::endl;
        return -1;
    }

    // Struct pointers for easier handling
    Bbox* const p_bbox = reinterpret_cast<Bbox*>(bbox_data);
    Probs* const p_class = reinterpret_cast<Probs*>(class_data);
    Landmark5* const p_landmark = reinterpret_cast<Landmark5*>(landmark_data);

    float detection_threshold = detectionParams.perClassThreshold[0];

    std::list<IndexWithProbability> final_objects;

    // Add the objects that pass the detection threshold to a list
    for(int i = 0; i < itr_bbox->inferDims.d[0]; i++)
    {
        if( p_class[i].class2_confidence > detection_threshold )
        {
           final_objects.emplace_back(i, p_class[i].class2_confidence);
        }
    }

    // NMS (Non-Maximum Suppression)
    NMS(final_objects, p_bbox, NMS_IOU_THRESHOLD);
    
    // Add the bboxes that passed the NMS to the metadata
    for(IndexWithProbability& elem: final_objects)
    {
        float x1 = p_bbox[elem.index].top_left.x * networkInfo.width;
        float y1 = p_bbox[elem.index].top_left.y * networkInfo.width;
        float x2 = p_bbox[elem.index].bottom_right.x * networkInfo.width;
        float y2 = p_bbox[elem.index].bottom_right.y * networkInfo.width;
        // Make the bounding box larger, so that more content is captured
        float padding_x = (x2-x1)*PADDING_FACTOR;
        float padding_y = (y2-y1)*PADDING_FACTOR;
        x1 -= padding_x;
        x2 += padding_x;
        y1 -= padding_y;
        y2 += padding_y;

        NvDsInferParseObjectInfo oinfo;
        oinfo.classId = 0;
        oinfo.left    = static_cast<unsigned int>(x1);
        oinfo.top     = static_cast<unsigned int>(y1);
        oinfo.width   = static_cast<unsigned int>(x2-x1);
        oinfo.height  = static_cast<unsigned int>(y2-y1);
        oinfo.detectionConfidence = elem.probability;
        objectList.push_back(oinfo);

        // We can draw the landmarks for debugging purposes, normally this is turned off
        if constexpr (DRAW_LANDMARKS != 0)
        {
            for(unsigned int i=0; i < NR_LANDMARKS; i++)
            {
                NvDsInferParseObjectInfo oinfo;
                oinfo.classId = 1;
                oinfo.left = static_cast<unsigned int>(p_landmark[elem.index].point[i].x * networkInfo.width);
                oinfo.top = static_cast<unsigned int>(p_landmark[elem.index].point[i].y * networkInfo.height);
                oinfo.width = 1;
                oinfo.height = 1;
                oinfo.detectionConfidence = elem.probability;
                objectList.push_back(oinfo);
            }
        }
    }

    return true;
}

extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseRetinaface(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRetinaface);
