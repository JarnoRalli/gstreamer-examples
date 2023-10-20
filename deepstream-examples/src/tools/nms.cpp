#include <algorithm>
#include <iterator>

#include "nms.hpp"


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
        std::copy_if(index_list.begin(), index_list.end(), std::back_inserter(candidates), 
            [&candidate, &min_iou_threshold, p_bbox](IndexWithProbability& elem){return IoU(p_bbox[candidate.index], p_bbox[elem.index]) > min_iou_threshold;});

        // Erase the overlapping items from the index_list
        index_list.remove_if(
            [&candidate, &min_iou_threshold, p_bbox](IndexWithProbability& elem){return IoU(p_bbox[candidate.index], p_bbox[elem.index]) > min_iou_threshold;});

        // Add the candidate to the list of candidates
        candidates.push_back(candidate);

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