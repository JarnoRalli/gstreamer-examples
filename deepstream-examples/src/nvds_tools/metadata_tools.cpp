#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"

#include "metadata_tools.hpp"

void get_tensor_metadata(NvDsBatchMeta *batch_meta) {
    // Iterate through all frames in the batch
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        // Iterate through all the metadata associated with the frame
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;

            // Check if the metadata type matches the tensor output metadata
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
            {
                NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
                // Now you can access and process the tensor metadata as needed
            }
        }
    }
}
