#include <memory>
#include <cstring>

#include <cuda_runtime.h>

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"
#include "gstnvdsinfer.h"

#include "data_types.hpp"
#include "metadata_tools.hpp"
#include "nms.hpp"

static constexpr unsigned int MUXER_OUTPUT_WIDTH = 1920;
static constexpr unsigned int MUXER_OUTPUT_HEIGHT = 1080;
static constexpr unsigned int MUXER_BATCH_TIMEOUT_USEC =4000000;

constexpr float NMS_IOU_THRESHOLD = 0.2f;

static void release_landmark_meta(gpointer data, gpointer user_data)
{
    // NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    // if(user_meta->user_meta_data)
    // {
    //     g_free(user_meta->user_meta_data);
    //     user_meta->user_meta_data = NULL;
    // }
}

static gpointer copy_landmark_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    gchar *src_user_metadata = (gchar*)user_meta->user_meta_data;
    gchar *dst_user_metadata = (gchar*)g_malloc0(sizeof(Landmark5));
    memcpy(dst_user_metadata, src_user_metadata, sizeof(Landmark5));
    return (gpointer)dst_user_metadata;
}

NvDsUserMeta* createUserMeta(NvDsBatchMeta* batch_meta, Landmark5* landmarks)
{
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if(user_meta)
    {
        user_meta->user_meta_data = (void*)landmarks;
        user_meta->base_meta.meta_type = NVDSINFER_LANDMARKS_META;
        user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)copy_landmark_meta;
        user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_landmark_meta;
    }
    return user_meta;
}

void attachLandmarksToObjects(NvDsObjectMeta* object_meta, NvDsBatchMeta* batch_meta, Landmark5* landmarks)
{
    NvDsUserMeta* user_meta = createUserMeta(batch_meta, landmarks);
    if (user_meta) 
    {
        nvds_add_user_meta_to_obj(object_meta, user_meta);
    }
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

/* Pad-added signal callback */
static void pad_added_handler(GstElement *src, GstPad *new_pad, GstElement *sink_element) {
    GstPad *sink_pad = gst_element_get_static_pad(sink_element, "sink");
    GstPadLinkReturn ret;
    GstCaps *new_pad_caps = NULL;
    GstStructure *new_pad_struct = NULL;
    const gchar *new_pad_type = NULL;

    /* Check the new pad's type */
    new_pad_caps = gst_pad_get_current_caps(new_pad);
    new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);

    ret = gst_pad_link(new_pad, sink_pad);
    if(GST_PAD_LINK_FAILED(ret))
    {
        g_printerr("pad_added_handler: failed to link %s -> %s on pad type %s\n", 
            gst_element_get_name(src),
            gst_element_get_name(sink_element),
            gst_structure_get_name(new_pad_struct));
    }

    /* Unreference the new pad's caps, if we got them */
    if (new_pad_caps != NULL)
        gst_caps_unref(new_pad_caps);

    /* Unreference the sink pad */
    gst_object_unref(sink_pad);
}



static GstPadProbeReturn pgie_post_processing(GstPad *pad, GstPadProbeInfo* info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsMetaList *p_frame = NULL;
    NvDsMetaList *p_obj = NULL;
    NvDsMetaList *p_user = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    NvDsInferLayerInfo *bboxes_info = NULL;
    NvDsInferLayerInfo *landmarks_info = NULL;
    NvDsInferLayerInfo *probs_info = NULL;
    
    Bbox* p_bbox = nullptr;
    Probs* p_class = nullptr;
    Landmark5* p_landmark = nullptr;
    int bboxes_index = -1;
    int landmarks_index = -1;
    int probs_index = -1;

    std::list<IndexWithProbability> detected_objects;

    // Iterate through batch meta data
    for(p_frame = batch_meta->frame_meta_list; p_frame != NULL; p_frame = p_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(p_frame->data);
        Point2D network_scaling(frame_meta->pipeline_width, frame_meta->pipeline_width);

        // Iterate through frame user meta data
        for(p_user = frame_meta->frame_user_meta_list; p_user != NULL; p_user = p_user->next)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)p_user->data;
            NvDsInferTensorMeta *tensor_data = NULL;

            // Only operate on type NVDSINFER_TENSOR_OUTPUT_META
            if(user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
            {
                tensor_data = (NvDsInferTensorMeta *)user_meta->user_meta_data;

                // Iterate through the tensor data
                for(int i = 0; i < tensor_data->num_output_layers; ++i)
                {
                    if(strcmp(tensor_data->output_layers_info[i].layerName, "bboxes") == 0)
                    {
                        bboxes_info = &tensor_data->output_layers_info[i];
                        bboxes_index = i;
                    }
                    if(strcmp(tensor_data->output_layers_info[i].layerName, "landmarks") == 0)
                    {
                        landmarks_info = &tensor_data->output_layers_info[i];
                        landmarks_index = i;
                    }
                    if(strcmp(tensor_data->output_layers_info[i].layerName, "classes") == 0)
                    {
                        probs_info = &tensor_data->output_layers_info[i];
                        probs_index = i;
                    }
                }
            }

            // We did not find the required meta-data
            if(!bboxes_info || !landmarks_info || !probs_info)
            {
                return GST_PAD_PROBE_OK;
            }

            NvDsInferDims &bboxes_dims = bboxes_info->inferDims;
            NvDsInferDims &landmarks_dims = bboxes_info->inferDims;
            NvDsInferDims &probs_dims = bboxes_info->inferDims;

            // Struct pointers for easier handling
            p_bbox = reinterpret_cast<Bbox*>(tensor_data->out_buf_ptrs_host[bboxes_index]);
            p_class = reinterpret_cast<Probs*>(tensor_data->out_buf_ptrs_host[probs_index]);
            p_landmark = reinterpret_cast<Landmark5*>(tensor_data->out_buf_ptrs_host[landmarks_index]);

            // Add the objects that pass the detection threshold to a list
            for(int index = 0; index < bboxes_dims.d[0]; index++)
            {
                if( p_class[index].class2_confidence > 0.5f )
                {
                    detected_objects.emplace_back(index, p_class[index].class2_confidence);
                }
            }

            // NMS (Non-Maximum Suppression)
            NMS(detected_objects, p_bbox, NMS_IOU_THRESHOLD);

            // Add the bboxes that passed the NMS to the object meta-data
            for(IndexWithProbability& elem: detected_objects)
            {
                // Create new object meta
                NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
                if (!obj_meta)
                {
                    g_print("Error: Failed to acquire object meta from pool\n");
                    return GST_PAD_PROBE_OK;
                }
                obj_meta->class_id = 0;
                obj_meta->confidence =  elem.probability;
                
                float x1 = p_bbox[elem.index].top_left.x * frame_meta->pipeline_width;
                float y1 = p_bbox[elem.index].top_left.y * frame_meta->pipeline_width;
                float x2 = p_bbox[elem.index].bottom_right.x * frame_meta->pipeline_width;
                float y2 = p_bbox[elem.index].bottom_right.y * frame_meta->pipeline_width;
                
                obj_meta->detector_bbox_info.org_bbox_coords.top = y1;
                obj_meta->detector_bbox_info.org_bbox_coords.left = x1;
                obj_meta->detector_bbox_info.org_bbox_coords.width = (x2 - x1);
                obj_meta->detector_bbox_info.org_bbox_coords.height = (y2 - y1);
                obj_meta->rect_params.top = y1;
                obj_meta->rect_params.left = x1;
                obj_meta->rect_params.width = (x2 - x1);
                obj_meta->rect_params.height = (y2 - y1);
                obj_meta->rect_params.border_width = 2;
                obj_meta->rect_params.border_color.red = 1.0f;
                obj_meta->rect_params.border_color.blue = 0.0f;
                obj_meta->rect_params.border_color.green = 0.0f;
                obj_meta->rect_params.border_color.alpha = 1.0f;
    
                nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
                Landmark5 scaled_landmark = p_landmark[elem.index] * network_scaling;
                attachLandmarksToObjects(obj_meta, batch_meta, &scaled_landmark);
            }
        }

        // Draw landmarks
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

        for(IndexWithProbability& elem: detected_objects)
        {
            Landmark5 scaled_landmark = p_landmark[elem.index] * network_scaling;
            for(int landmark_nr = 0; landmark_nr < Landmark5::size; landmark_nr++)
            {
                if(!display_meta)
                {
                    g_print("Error: Failed to acquire display meta from pool\n");
                    return GST_PAD_PROBE_OK;
                }
                NvOSD_CircleParams &cparams = display_meta->circle_params[display_meta->num_circles++];
                cparams.xc = scaled_landmark.point[landmark_nr].x;
                cparams.yc = scaled_landmark.point[landmark_nr].y;
                cparams.radius = 1;
                
                cparams.circle_color.red = getColour(TenColours, landmark_nr).red;
                cparams.circle_color.blue = getColour(TenColours, landmark_nr).blue;
                cparams.circle_color.green = getColour(TenColours, landmark_nr).green;
                cparams.circle_color.alpha = getColour(TenColours, landmark_nr).alpha;
                nvds_add_display_meta_to_frame(frame_meta, display_meta);

                // If we have reached the maximum number of drawable elements per
                // display meta, acquire new one from the pool
                if(display_meta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
                {
                    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                } 
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

// https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation/blob/master/deepstream_pose_estimation_app.cpp
int main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstCaps *caps = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstElement *source = NULL, *demuxer = NULL, *queue1 = NULL, *h264parser = NULL, *h264decoder = NULL, *streammux = NULL, *pgie = NULL,
               *videoconvert = NULL, *osd = NULL, *sink = NULL, *pipeline = NULL, *queue2 = NULL;

    g_print("Source file: %s\n", argv[1]);

    // Initialize GStreamer
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    // Create the elements
    if (!(pipeline = gst_pipeline_new("deepstream_retinaface")))
    {
        g_printerr("pipeline could not be created");
        return -1;
    }

    // Create a source element
    if (!(source = gst_element_factory_make("filesrc", "filesource")))
    {
        g_printerr("source could not be created");
        return -1;
    }

    // Create a demuxer element
    if (!(demuxer = gst_element_factory_make("qtdemux", "demuxer")))
    {
        g_printerr("demuxer could not be created");
        return -1;
    }

    // Create a queue
    if (!(queue1 = gst_element_factory_make("queue", "queue1")))
    {
        g_printerr("queue1 could not be created");
        return -1;
    }

    // Create a h264parser
    if (!(h264parser = gst_element_factory_make("h264parse", "h264parser")))
    {
        g_printerr("h264parser could not be created");
        return -1;
    }

    // Create a h264decoder
    if (!(h264decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2decoder")))
    {
        g_printerr("h264decoder could not be created");
        return -1;
    }

    // Create a streammux
    if (!(streammux = gst_element_factory_make("nvstreammux", "nvstreammux")))
    {
        g_printerr("streammuxer could not be created");
        return -1;
    }

    // Create an inference element
    if (!(pgie = gst_element_factory_make("nvinfer", "inference")))
    {
        g_printerr("pgie could not be created");
        return -1;
    }

    // Create a videoconverter
    if (!(videoconvert = gst_element_factory_make("nvvideoconvert", "videoconvert")))
    {
        g_printerr("videconverter could not be created");
        return -1;
    }

    // Create a nvdsosd for drawing bounding boxes etc
    if (!(osd = gst_element_factory_make("nvdsosd", "nvdsosd")))
    {
        g_printerr("osd could not be created");
        return -1;
    }

    // Create a queue2
    if (!(queue2 = gst_element_factory_make("queue", "queue2")))
    {
        g_printerr("queue2 could not be created");
        return -1;
    }

    // Create a sink
    if (!(sink = gst_element_factory_make("nveglglessink", "sink")))
    {
        g_printerr("sink could not be created");
        return -1;
    }

    // Set the input file location
    g_object_set(G_OBJECT(source), "location", argv[1], NULL);

    // Set streammux properties
    g_object_set(G_OBJECT(streammux), 
        "width", MUXER_OUTPUT_WIDTH,
        "height", MUXER_OUTPUT_HEIGHT,
        "batch-size", 1,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
    
    // Set pgie properties
    g_object_set(G_OBJECT(pgie), "config-file-path", "/media/870_EVO_2TB/projects/gstreamer-examples/deepstream-examples/deepstream-retinaface/config_detector_dummy_parser.txt", NULL);

    // Add a message handler
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // Add elements to the pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, demuxer, queue1, h264parser, h264decoder, streammux, pgie, videoconvert, osd, queue2, sink, NULL);

    // --- LINK ELEMENTS ---
    // source -> demuxer -> queue1 -> h264parser -> h264decoder -> streammux -> pgie -> videoconvert -> osd -> queue2 -> sink

    // Link source -> demuxer
    if(!gst_element_link_many(source, demuxer, NULL))
    {
        g_printerr("Failed to link: source -> demuxer\n");
        return -1;
    }

    // Link demuxer -> queue1 dynamically
    g_signal_connect(demuxer, "pad-added", G_CALLBACK(pad_added_handler), queue1);

    // Link queue1 -> h264parser -> h264decoder
    if(!gst_element_link_many(queue1, h264parser, h264decoder, NULL))
    {
        g_printerr("Failed to link: queue1 -> h264parser -> h264decoder\n");
        return -1;
    }
        
    // Link decoder to streammux
    GstPad *sinkpad = NULL, *srcpad = NULL;
    if(!(srcpad = gst_element_get_static_pad(h264decoder, "src")))
    {
        g_printerr("Failed to get decoder source pad\n");
        return -1;
    }
    
    if(!(sinkpad = gst_element_get_request_pad(streammux, "sink_0")))
    {
        g_printerr("Failed to get streammux sink pad\n");
        return -1;
    }

    if(gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link decoder to streammux\n");
        return -1;
    }

    // Link streammux -> pgie -> videoconvert -> osd -> queue2 -> sink
    if(!gst_element_link_many(streammux, pgie, videoconvert, osd, queue2, sink, NULL))
    {
        g_printerr("Failed to link: streammux -> pgie -> videoconvert -> osd -> queue2 -> sink\n");
        return -1;
    }

    // --- ADD PROBES ---
    
    GstPad *pgie_src_pad = NULL;
    if(!(pgie_src_pad = gst_element_get_static_pad(pgie, "src")))
    {
        g_printerr("Could not get pgie src pad");
        return -1;
    }else{
        gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_post_processing, NULL, NULL);
    }    

    // Set the pipeline to playing state
    g_print("Playing file: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Loop until EOS or error
    g_main_loop_run(loop);

    // Clean once the main loop has been finished
    g_print("Stopped playing");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}