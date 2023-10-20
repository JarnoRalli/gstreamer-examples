#pragma once

#include "gstnvdsmeta.h"

#define NVDSINFER_LANDMARKS_META (nvds_get_user_meta_type(const_cast<gchar*>("EINHERJAR.NVINFER.USER_META")))

struct _NvDsBatchMeta;
typedef _NvDsBatchMeta NvDsBatchMeta;

void get_tensor_metadata(NvDsBatchMeta *batch_meta);
