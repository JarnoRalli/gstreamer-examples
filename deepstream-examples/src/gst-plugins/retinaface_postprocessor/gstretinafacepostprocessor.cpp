/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025, Jarno Ralli
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file retinaface_postprocessor_plugin.cpp
 * @brief GStreamer plugin definition for the RetinaFace post-processor element.
 *
 * This plugin registers the `retinaface_postprocessor` GStreamer element, which performs
 * post-processing for RetinaFace detection results. It can be used to filter, normalize,
 * or otherwise process bounding box outputs from neural networks.
 *
 * The plugin is initialized and registered with GStreamer through the GST_PLUGIN_DEFINE macro,
 * which defines the plugin entry point and metadata.
 *
 * @see gst_retinafacepostprocessor_init()
 * @see GST_PLUGIN_DEFINE
 */

/*
 * SECTION:element-retinaface_postprocessor
 * @short_description: Post-processes RetinaFace detection outputs.
 *
 * The `retinaface_postprocessor` element takes raw detection outputs (e.g., bounding boxes and landmarks)
 * from a neural network and processes them into structured face detection results.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v -m fakesrc ! retinaface_postprocessor ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>

#include "gstretinafacepostprocessor.hpp"

// TODO: these need to be moved to a different place
#include <iostream>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"

GST_DEBUG_CATEGORY_STATIC(gst_retinafacepostprocessor_debug);
#define GST_CAT_DEFAULT gst_retinafacepostprocessor_debug

/* Filter signals and args */
enum
{
    /* FILL ME */
    LAST_SIGNAL
};

enum
{
    PROP_0,
    PROP_SILENT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("ANY"));

static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("ANY"));

#define gst_retinafacepostprocessor_parent_class parent_class
G_DEFINE_TYPE(Gstretinafacepostprocessor, gst_retinafacepostprocessor, GST_TYPE_ELEMENT);

static void gst_retinafacepostprocessor_set_property(GObject *object, guint prop_id, const GValue *value,
                                                     GParamSpec *pspec);
static void gst_retinafacepostprocessor_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);

static gboolean      gst_retinafacepostprocessor_sink_event(GstPad *pad, GstObject *parent, GstEvent *event);
static GstFlowReturn gst_retinafacepostprocessor_chain(GstPad *pad, GstObject *parent, GstBuffer *buf);

/* GObject vmethod implementations */

/**
 * @brief Initializes the Gstretinafacepostprocessor class.
 *
 * This function sets up the class structure for the Gstretinafacepostprocessor
 * element. It typically registers class-specific properties, signals,
 * and virtual function pointers required for the GStreamer element’s
 * operation.
 *
 * This function is called once during the plugin loading process to
 * configure the class-level data.
 *
 * @param klass A pointer to the GstretinafacepostprocessorClass structure to initialize.
 */
static void gst_retinafacepostprocessor_class_init(GstretinafacepostprocessorClass *klass)
{
    GObjectClass    *gobject_class;
    GstElementClass *gstelement_class;

    gobject_class    = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;

    gobject_class->set_property = gst_retinafacepostprocessor_set_property;
    gobject_class->get_property = gst_retinafacepostprocessor_get_property;

    g_object_class_install_property(
        gobject_class, PROP_SILENT,
        g_param_spec_boolean("silent", "Silent", "Produce verbose output ?", FALSE, G_PARAM_READWRITE));

    gst_element_class_set_details_simple(gstelement_class, "RetinaFace Post-processor", "Filter/Video",
                                         "RetinaFace Post-processor moves bbox and landmarks from CUDA to host memory, "
                                         "applies IOU and makes the data available in meta-data",
                                         "Jarno Ralli <jarno@ralli.fi>");

    gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&src_factory));
    gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&sink_factory));
}

/**
 * @brief Initializes the Gstretinafacepostprocessor element.
 *
 * This function is responsible for setting up the newly created
 * Gstretinafacepostprocessor instance. It performs the following tasks:
 * - Initializes the element.
 * - Instantiates source and sink pads and adds them to the element.
 * - Sets callback functions for pad events and data handling.
 * - Initializes any custom data structures or states within the instance.
 *
 * @param filter A pointer to the Gstretinafacepostprocessor instance being initialized.
 */
static void gst_retinafacepostprocessor_init(Gstretinafacepostprocessor *filter)
{
    filter->sinkpad = gst_pad_new_from_static_template(&sink_factory, "sink");
    gst_pad_set_event_function(filter->sinkpad, GST_DEBUG_FUNCPTR(gst_retinafacepostprocessor_sink_event));
    gst_pad_set_chain_function(filter->sinkpad, GST_DEBUG_FUNCPTR(gst_retinafacepostprocessor_chain));
    GST_PAD_SET_PROXY_CAPS(filter->sinkpad);
    gst_element_add_pad(GST_ELEMENT(filter), filter->sinkpad);

    filter->srcpad = gst_pad_new_from_static_template(&src_factory, "src");
    GST_PAD_SET_PROXY_CAPS(filter->srcpad);
    gst_element_add_pad(GST_ELEMENT(filter), filter->srcpad);

    filter->silent = FALSE;
}

/**
 * @brief Starts the RetinaFace postprocessor element.
 *
 * This function is called when the GStreamer element transitions to the READY or PAUSED state.
 * It is typically used to allocate memory or initialize resources needed for processing frames
 * during the element's lifetime.
 *
 * @param element A pointer to the GstElement, cast internally to Gstretinafacepostprocessor.
 * @return TRUE if the start-up was successful, FALSE otherwise.
 *
 * This is where you should allocate persistent memory that is needed between consecutive frames.
 * If allocation fails, return FALSE to signal an error and prevent the pipeline from continuing.
 */
static gboolean gst_retinafacepostprocessor_start(GstElement *element)
{
    Gstretinafacepostprocessor *self = GST_RETINAFACE_POSTPROCESSOR(element);
}

/**
 * @brief Stops the RetinaFace postprocessor element.
 *
 * This function is called when the GStreamer element transitions from READY to NULL.
 * It should release any memory or resources allocated in gst_retinafacepostprocessor_start().
 *
 * @param element A pointer to the GstElement, cast internally to Gstretinafacepostprocessor.
 * @return TRUE on successful cleanup, FALSE otherwise.
 *
 * All runtime memory or resources allocated during start() should be freed here to avoid leaks.
 * This ensures that the element is cleanly reset or destroyed when removed from the pipeline.
 */
static gboolean gst_retinafacepostprocessor_stop(GstElement *element)
{
    Gstretinafacepostprocessor *self = GST_RETINAFACE_POSTPROCESSOR(element);
    return TRUE;
}

/**
 * @brief Sets the value of a property on the retinaface_postprocessor element.
 *
 * This function is called by the GObject property system when an external component
 * sets a property on the `retinaface_postprocessor` element. The function updates
 * the internal state of the element based on the provided property ID and value.
 *
 * The supported properties are typically registered during class initialization using
 * g_object_class_install_property().
 *
 * @param object The GObject instance (cast to GstRetinafacePostprocessor*).
 * @param prop_id The numeric ID of the property to set.
 * @param value A GValue containing the new value for the property.
 * @param pspec The GParamSpec describing the property.
 *
 * @see g_object_set_property()
 * @see g_object_class_install_property()
 * @see gst_retinafacepostprocessor_get_property()
 */
static void gst_retinafacepostprocessor_set_property(GObject *object, guint prop_id, const GValue *value,
                                                     GParamSpec *pspec)
{
    Gstretinafacepostprocessor *filter = GST_RETINAFACE_POSTPROCESSOR(object);

    switch (prop_id)
    {
        case PROP_SILENT:
            filter->silent = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/**
 * @brief Retrieves the value of a property from the retinaface_postprocessor element.
 *
 * This function is called by the GObject property system when an external component
 * requests the value of a property on the `retinaface_postprocessor` element.
 * It reads the internal state of the element and writes the corresponding value to `value`.
 *
 * The function handles various property IDs defined during element class initialization
 * (e.g., threshold, scale factor).
 *
 * @param object The GObject instance (cast to GstRetinafacePostprocessor*).
 * @param prop_id The numeric ID of the property to retrieve.
 * @param value A GValue to store the retrieved property value.
 * @param pspec The GParamSpec describing the property.
 *
 * @see g_object_get_property()
 * @see g_object_class_install_property()
 * @see gst_retinafacepostprocessor_set_property()
 */
static void gst_retinafacepostprocessor_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    Gstretinafacepostprocessor *filter = GST_RETINAFACE_POSTPROCESSOR(object);

    switch (prop_id)
    {
        case PROP_SILENT:
            g_value_set_boolean(value, filter->silent);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* GstElement vmethod implementations */

/**
 * @brief Handles events received on the sink pad of the retinaface_postprocessor element.
 *
 * This function processes incoming events (e.g., EOS, CAPS, FLUSH) on the sink pad.
 * It allows the element to respond appropriately to stream control signals, such as
 * end-of-stream notifications, format negotiations, or flushing.
 *
 * The function typically forwards events downstream, filters them, or takes specific
 * action based on the event type.
 *
 * @param pad The sink pad receiving the event.
 * @param parent The parent GstElement (typically cast to GstRetinafacePostprocessor*).
 * @param event The incoming event to be handled.
 *
 * @return TRUE if the event was successfully handled (or forwarded); FALSE otherwise.
 *
 * @see gst_pad_event_function()
 * @see gst_pad_push_event()
 */
static gboolean gst_retinafacepostprocessor_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
    Gstretinafacepostprocessor *filter;
    gboolean                    ret;

    filter = GST_RETINAFACE_POSTPROCESSOR(parent);

    GST_LOG_OBJECT(filter, "Received %s event: %" GST_PTR_FORMAT, GST_EVENT_TYPE_NAME(event), event);

    switch (GST_EVENT_TYPE(event))
    {
        case GST_EVENT_CAPS: {
            GstCaps *caps;

            gst_event_parse_caps(event, &caps);
            /* do something with the caps */

            /* and forward */
            ret = gst_pad_event_default(pad, parent, event);
            break;
        }
        default:
            ret = gst_pad_event_default(pad, parent, event);
            break;
    }
    return ret;
}

/**
 * @brief Processes incoming buffers on the sink pad of the retinaface_postprocessor element.
 *
 * This function is the chain function for the element's sink pad. It is called whenever
 * a new buffer arrives. The function handles buffer processing, which may include parsing
 * neural network output data, applying post-processing logic, and forwarding the processed
 * results downstream.
 *
 * @param pad The sink pad that received the buffer.
 * @param parent The parent GstElement (cast to GstRetinafacePostprocessor* if needed).
 * @param buf The buffer containing incoming data to be processed.
 *
 * @return GST_FLOW_OK if processing succeeds and the buffer is pushed downstream,
 *         or an appropriate GstFlowReturn error code on failure.
 *
 * @see gst_pad_chain_function()
 */
static GstFlowReturn gst_retinafacepostprocessor_chain(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    Gstretinafacepostprocessor *filter;
    NvDsMetaList               *list_frame = NULL;

    filter = GST_RETINAFACE_POSTPROCESSOR(parent);

    if (filter->silent == FALSE)
        g_print("I'm plugged, therefore I'm in.\n");

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (batch_meta)
    {
        // Iterate over frame meta
        for (list_frame = batch_meta->frame_meta_list; list_frame != NULL; list_frame = list_frame->next)
        {
            NvDsFrameMeta    *frame_meta     = (NvDsFrameMeta *)(list_frame->data);
            NvDsUserMetaList *list_user_meta = NULL;

            // Iterate over frame_user_meta
            for (list_user_meta = frame_meta->frame_user_meta_list; list_user_meta != NULL;
                 list_user_meta = list_user_meta->next)
            {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)list_user_meta->data;

                // We have tensor data that has been attached by infer element
                if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
                {
                    NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;

                    // Iterate over the output layers from the model
                    for (int i = 0; i < tensor_meta->num_output_layers; i++)
                    {
                        NvDsInferLayerInfo layer_info = tensor_meta->output_layers_info[i];
                        std::cout << "Layer[" << i << "] name" << layer_info.layerName << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        g_printerr("No batch metadata available.\n");
    }

    /* just push out the incoming buffer without touching it */
    return gst_pad_push(filter->srcpad, buf);
}

/**
 * @brief Initializes the RetinaFace post-processor GStreamer plugin.
 *
 * This function is the entry point called by GStreamer when the plugin is loaded.
 * It registers the `retinaface_postprocessor` element and sets up any necessary
 * features, such as debugging categories or element factories.
 *
 * It is passed to the GST_PLUGIN_DEFINE macro and invoked automatically by
 * the GStreamer plugin loader.
 *
 * @param retinaface_postprocessor The GstPlugin structure representing this plugin instance.
 *
 * @return TRUE if initialization succeeds and the element is registered; FALSE on failure.
 *
 * @see GST_PLUGIN_DEFINE
 * @see gst_element_register()
 * @see GST_DEBUG_CATEGORY_INIT
 */
static gboolean retinaface_postprocessor_init(GstPlugin *retinaface_postprocessor)
{
    /* debug category for filtering log messages
     *
     */
    GST_DEBUG_CATEGORY_INIT(gst_retinafacepostprocessor_debug, "retinaface_postprocessor", 0,
                            "retinaface_postprocessor");

    return gst_element_register(retinaface_postprocessor, "retinaface_postprocessor", GST_RANK_NONE,
                                GST_TYPE_RETINAFACE_POSTPROCESSOR);
}

/* PACKAGE: this is usually set by meson depending on some _INIT macro
 * in meson.build and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use meson to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirstretinaface_postprocessor"
#endif

/**
 * @brief Registers the RetinaFace post-processor plugin with GStreamer.
 *
 * This macro defines the plugin entry point, version, license, and other metadata required
 * by GStreamer. The plugin will be discoverable and usable under the name "retinaface_postprocessor".
 */
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, retinaface_postprocessor, "RetinaFace Post-processor",
                  retinaface_postprocessor_init, PACKAGE_VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
