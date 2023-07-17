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
// 1. Add landmarks to user-type meta-data
//  - https://docs.nvidia.com/metropolis/deepstream/sdk-api/group__gstreamer__metagroup__api.html#ga491a88faec97ebca5742facfa80c5e6a
//  - https://docs.nvidia.com/metropolis/deepstream/5.0/dev-guide/index.html#page/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_metadata.html
// 2. Batch handling, currently only handles a batch size of 1
// 3. Proper error handling


#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

// It appears that only a very limited number of parameters can be passed to this function
constexpr float NMS_IOU_THRESHOLD = 0.2f;
// Percentage of padding added around the bounding boxes
constexpr float PADDING_FACTOR = 0.1f;
// Number of landmark points
constexpr unsigned int NR_LANDMARKS = 5;
// Draw landmarks
// You need to modify the nvinfer configuration so that the expected number of classes is 2 in order for this to: num-detected-classes=2
constexpr unsigned int DRAW_LANDMARKS = 0;

extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

/**
 * @class Point2D
 * @brief A struct to hold 2D points.
 */
struct alignas(float) Point2D
{
    Point2D() : x(0.0f), y(0.0f)
    {}

    Point2D(float x_in, float y_in) : x(x_in), y(y_in)
    {}

    float x;
    float y;
};

/**
 * @struct Bbox
 * @brief A struct that defines a bounding box.
 */
struct alignas(float) Bbox
{
    Bbox() : top_left(Point2D()), bottom_right(Point2D())
    {}
    
    Bbox(Point2D top_left_in, Point2D bottom_right_in) : top_left(top_left_in), bottom_right(bottom_right_in)
    {}

    /**
     * @brief Returns area of the bbox.
     * @return area of the bouding box.
     */
    float getArea() const
    {
        return (bottom_right.x - top_left.x) * (bottom_right.y - top_left.y);
    }

    Point2D top_left;
    Point2D bottom_right;
};

/**
 * @struct Landmark
 * @brief A struct that defines landmarks
 */
struct alignas(float) Landmark
{
    Point2D point[NR_LANDMARKS];
};

/**
 * @struct Bbox
 * @brief A struct that defines the class confidences/probabilities.
 */
struct alignas(float) Class{
    float class1_confidence;
    float class2_confidence;
};

/**
 * @struct IndexWithProbability
 * @brief A struct that contains index and a confidence/probability score.
 */
struct IndexWithProbability
{
    IndexWithProbability() : index(0), probability(-1.0f)
    {}

    IndexWithProbability(std::size_t index_in, float probability_in) : index(index_in), probability(probability_in)
    {}

    std::size_t index;
    float probability;
};

/**
 * @brief Calculates intersection over union for given bboxes.
 * @param[in] a : bbox.
 * @param[in] b : bbox.
 * @return IoU
 */
float IoU(Bbox const& a, Bbox const& b)
{
    // Minimum coordinates of bottom-right points
    float x_max = std::min(a.bottom_right.x, b.bottom_right.x);
    float y_max = std::min(a.bottom_right.y, b.bottom_right.y);

    // Maximum coordinates of top-left point
    float x_min = std::max(a.top_left.x, b.top_left.x);
    float y_min = std::max(a.top_left.y, b.top_left.y);
    
    // Width and height for intersection
    float w = std::max(0.0f, x_max - x_min);
    float h = std::max(0.0f, y_max - y_min);

    float intersection_area = w * h;
    float union_area = a.getArea() + b.getArea() - intersection_area;

    return intersection_area / union_area;
}

/**
 * @class Custom Less operator for sorting IndexWithProbability-objects using probability.
 */
struct
{
    bool operator()(IndexWithProbability &a, IndexWithProbability &b) const {return a.probability < b.probability;}
}customLessOperator;


/**
 * @brief Non-Minimum Suppression
 * @param[in/out] index_list : a list of objects, that contain indices to bboxes in p_bbox, that will be filtered using NMS.
 * @param[in] p_bbox : a pointer to beginning of an array where the bboxes reside.
 * @param[in] min_iou_threshold : IoU above this threshold will be filtered.
 * @return number of elements left in the index_list after NMS.
 */
std::size_t NMS(std::list<IndexWithProbability>& index_list, Bbox* p_bbox, float min_iou_threshold)
{
    if(index_list.size() == 0)
    {
        return 0;
    }

    std::list<IndexWithProbability> results;
    std::list<IndexWithProbability> candidates;

    IndexWithProbability candidate;

    while(index_list.size()>0)
    {
        candidates.clear();

        // First element is the candidate for IoU matching
        candidate = index_list.front();
        index_list.pop_front();

        // Add those bounding boxes that overlap enough with the candidate to the list of candidates
        std::copy_if(index_list.begin(), index_list.end(), candidates.end(), 
            [&candidate, &min_iou_threshold, p_bbox](IndexWithProbability& elem){return IoU(p_bbox[candidate.index], p_bbox[elem.index]) > min_iou_threshold;});
        // Erase the overlapping items from the index_list
        std::erase_if(index_list,
            [&candidate, &min_iou_threshold, p_bbox](IndexWithProbability& elem){return IoU(p_bbox[candidate.index], p_bbox[elem.index]) > min_iou_threshold;});
        // Sort the candidates, in ascending order, based on the detection probability
        candidates.sort(customLessOperator);
        // Store the one with highest probability
        results.push_back(candidates.back());
    }

    index_list = results;
    return results.size();
}

std::ostream &operator<<(std::ostream &os, Bbox const &bbox)
{
    os << "Top-left: (" << bbox.top_left.x << ", " << bbox.top_left.y << ")" << std::endl;
    os << "Bottom-right: (" << bbox.bottom_right.x << ", " << bbox.bottom_right.y << ")" << std::endl;
    return os;
}

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
    Class* const p_class = reinterpret_cast<Class*>(class_data);
    Landmark* const p_landmark = reinterpret_cast<Landmark*>(landmark_data);

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
