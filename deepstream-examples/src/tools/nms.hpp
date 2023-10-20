#include <list>
#include <iostream>

#include "data_types.hpp"

/**
 * @brief Calculates intersection over union for given bboxes.
 * @param[in] a : bbox.
 * @param[in] b : bbox.
 * @return IoU
 */
float IoU(Bbox const& a, Bbox const& b);

/**
 * @brief Non-Minimum Suppression
 * @param[in/out] index_list : a list of objects, that contain indices to bboxes in p_bbox, that will be filtered using NMS.
 * @param[in] p_bbox : a pointer to beginning of an array where the bboxes reside.
 * @param[in] min_iou_threshold : IoU above this threshold will be filtered.
 * @return number of elements left in the index_list after NMS.
 */
std::size_t NMS(std::list<IndexWithProbability>& index_list, Bbox* p_bbox, float min_iou_threshold);

std::ostream &operator<<(std::ostream &os, Bbox const &bbox);