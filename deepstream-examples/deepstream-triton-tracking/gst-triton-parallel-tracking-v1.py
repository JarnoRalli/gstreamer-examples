"""
This file implements a simple pipeline with detector and classifiers, using Triton Inference Server
for doing inference. Several streams can be processed using the pipeline.

This example uses a probe attached to the osd element in order to modify the way how the detections
are drawn to the video. Following objects are detected:

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

This version draws information of objects that are further away from the camera first.
Everything else being same, due to projective geometry, objects that are smaller are further
away from the camera. By drawing bounding boxes and labels for objects that are further
away from the camera first, object IDs and labels will be easier to read.

For more information regarding the input parameters, execute the following:
>> python3 gst-triton-parallel-tracking-v1.py -h

In order to process a single video file:
>> python3 gst-triton-tracking-v1.py -u /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

In order to process several video files:
>> python3 gst-triton-tracking-v1.py -u \
  /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4,/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4
"""

import platform
from urllib.parse import urlparse
from collections import namedtuple
from operator import attrgetter
import argparse
import configparser
import os
import sys
import signal
import pyds
from helpers import gsthelpers
import gi
import logging
from typing import Any

logger = logging.getLogger(__name__)

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib, GObject  # noqa: E402, F401

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
past_tracking_meta = [0]

MetaObject = namedtuple(
    "MetaObject",
    ["left", "top", "height", "width", "area", "bottom", "id", "text", "class_id"],
)

ColorObject = namedtuple("ColorObject", ["red", "green", "blue", "alpha"])

ColorList = {
    PGIE_CLASS_ID_VEHICLE: ColorObject(red=1.0, green=0.0, blue=0.0, alpha=1.0),
    PGIE_CLASS_ID_BICYCLE: ColorObject(red=0.0, green=1.0, blue=0.0, alpha=1.0),
    PGIE_CLASS_ID_PERSON: ColorObject(red=0.0, green=0.0, blue=1.0, alpha=1.0),
    PGIE_CLASS_ID_ROADSIGN: ColorObject(red=1.0, green=0.0, blue=1.0, alpha=1.0),
}


def osd_sink_pad_buffer_probe(
    pad: Gst.Pad, info: Gst.PadProbeIndo, u_data: Any
) -> Gst.PadProbeReturn:
    frame_number = 0
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0,
        PGIE_CLASS_ID_PERSON: 0,
        PGIE_CLASS_ID_BICYCLE: 0,
        PGIE_CLASS_ID_ROADSIGN: 0,
    }
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    meta_list = []

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_counter[obj_meta.class_id] += 1

            obj = MetaObject(
                left=obj_meta.tracker_bbox_info.org_bbox_coords.left,
                top=obj_meta.tracker_bbox_info.org_bbox_coords.top,
                height=obj_meta.tracker_bbox_info.org_bbox_coords.height,
                width=obj_meta.tracker_bbox_info.org_bbox_coords.width,
                area=obj_meta.tracker_bbox_info.org_bbox_coords.height
                * obj_meta.tracker_bbox_info.org_bbox_coords.width,
                bottom=obj_meta.tracker_bbox_info.org_bbox_coords.top
                + obj_meta.tracker_bbox_info.org_bbox_coords.height,
                id=obj_meta.object_id,
                text=f"ID: {obj_meta.object_id:04d}, Class: {pyds.get_string(obj_meta.text_params.display_text)}",
                class_id=obj_meta.class_id,
            )
            meta_list.append(obj)

            obj_meta.text_params.display_text = ""
            obj_meta.text_params.set_bg_clr = 0
            obj_meta.rect_params.border_width = 0

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        meta_list_sorted = sorted(meta_list, key=attrgetter("bottom"))
        max_labels = 10  # Define a suitable number for max_labels
        num_objects = len(meta_list_sorted)
        num_meta_objects = (num_objects + max_labels - 1) // max_labels

        # Create a single display_meta for the frame counter and object counts
        display_meta_main = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta_main.num_labels = 1
        py_nvosd_text_params = display_meta_main.text_params[0]
        py_nvosd_text_params.display_text = (
            f"Frame Number={frame_number:05d}, Number of Objects={num_rects:04d}, "
            f"Vehicles={obj_counter[PGIE_CLASS_ID_VEHICLE]:04d}, "
            f"Persons={obj_counter[PGIE_CLASS_ID_PERSON]:04d}, "
            f"Bicycles={obj_counter[PGIE_CLASS_ID_BICYCLE]:04d}, "
            f"Road Signs={obj_counter[PGIE_CLASS_ID_ROADSIGN]:04d}"
        )
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta_main)

        for i in range(num_meta_objects):
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 0

            start_idx = i * max_labels
            end_idx = min((i + 1) * max_labels, num_objects)

            for j, idx in enumerate(range(start_idx, end_idx)):
                x = int(meta_list_sorted[idx].left)
                y = int(meta_list_sorted[idx].top) - 15

                if x < 0 or y < 0:
                    continue

                display_meta.text_params[
                    display_meta.num_labels
                ].display_text = meta_list_sorted[idx].text
                display_meta.text_params[display_meta.num_labels].x_offset = x
                display_meta.text_params[display_meta.num_labels].y_offset = y
                display_meta.text_params[
                    display_meta.num_labels
                ].font_params.font_name = "Serif"
                display_meta.text_params[
                    display_meta.num_labels
                ].font_params.font_size = 10
                display_meta.text_params[
                    display_meta.num_labels
                ].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                display_meta.text_params[display_meta.num_labels].set_bg_clr = 1
                display_meta.text_params[display_meta.num_labels].text_bg_clr.set(
                    0.45, 0.20, 0.50, 0.75
                )
                display_meta.num_labels += 1

            display_meta.num_rects = end_idx - start_idx
            for j, idx in enumerate(range(start_idx, end_idx)):
                red = ColorList[meta_list_sorted[idx].class_id].red
                green = ColorList[meta_list_sorted[idx].class_id].green
                blue = ColorList[meta_list_sorted[idx].class_id].blue
                alpha = ColorList[meta_list_sorted[idx].class_id].alpha

                display_meta.rect_params[j].left = meta_list_sorted[idx].left
                display_meta.rect_params[j].top = meta_list_sorted[idx].top
                display_meta.rect_params[j].width = meta_list_sorted[idx].width
                display_meta.rect_params[j].height = meta_list_sorted[idx].height
                display_meta.rect_params[j].border_width = 2
                display_meta.rect_params[j].border_color.red = red
                display_meta.rect_params[j].border_color.green = green
                display_meta.rect_params[j].border_color.blue = blue
                display_meta.rect_params[j].border_color.alpha = alpha
                display_meta.rect_params[j].has_bg_color = 0

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Past tracking meta data
    if past_tracking_meta[0] == 1:
        l_user = batch_meta.batch_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if (
                user_meta
                and user_meta.base_meta.meta_type
                == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META
            ):
                try:
                    pPastFrameObjBatch = pyds.NvDsPastFrameObjBatch.cast(
                        user_meta.user_meta_data
                    )
                except StopIteration:
                    break
                for trackobj in pyds.NvDsPastFrameObjBatch.list(pPastFrameObjBatch):
                    print("streamId=", trackobj.streamID)
                    print("surfaceStreamID=", trackobj.surfaceStreamID)
                    for pastframeobj in pyds.NvDsPastFrameObjStream.list(trackobj):
                        print("numobj=", pastframeobj.numObj)
                        print("uniqueId=", pastframeobj.uniqueId)
                        print("classId=", pastframeobj.classId)
                        print("objLabel=", pastframeobj.objLabel)
                        for objlist in pyds.NvDsPastFrameObjList.list(pastframeobj):
                            print("frameNum:", objlist.frameNum)
                            print("tBbox.left:", objlist.tBbox.left)
                            print("tBbox.width:", objlist.tBbox.width)
                            print("tBbox.top:", objlist.tBbox.top)
                            print("tBbox.right:", objlist.tBbox.height)
                            print("confidence:", objlist.confidence)
                            print("age:", objlist.age)
            try:
                l_user = l_user.next
            except StopIteration:
                break

    return Gst.PadProbeReturn.OK


class MultiPlayer:
    def __init__(self, uri_list: str):
        """MultiPlayer constructor.

        Parameters
        ----------
        uri_list : str
            A comma separated list of URIs
        """
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.bin_cntr = 0

        # Register signal handlers
        signal.signal(signal.SIGINT, self.stop_handler)
        signal.signal(signal.SIGTERM, self.stop_handler)
        signal.signal(signal.SIGHUP, self.stop_handler)

        # If the uri_list contains files, check that these exist
        for uri in uri_list.split(","):
            uri_parsed = urlparse(uri)
            if uri_parsed.scheme == "file":
                if not os.path.exists(uri_parsed.path):
                    logger.error(f"File '{uri_parsed.path}' does not exist")
                    sys.exit(-1)

        # Create an empty pipeline
        self.pipeline = Gst.Pipeline.new("input-pipeline")
        assert self.pipeline is not None

        # Create elements
        nvmultiurisrcbin = gsthelpers.create_element(
            "nvmultiurisrcbin", "multiurisrcbin"
        )
        demuxer = gsthelpers.create_element("nvstreamdemux", "demuxer")

        # Add elements to the pipeline
        self.pipeline.add(nvmultiurisrcbin)
        self.pipeline.add(demuxer)

        # Set the multiurisrcbin properties
        logger.info(f"URI-list: {uri_list}")
        nvmultiurisrcbin.set_property("uri-list", uri_list)
        nvmultiurisrcbin.set_property("width", 1920)
        nvmultiurisrcbin.set_property("height", 1080)
        nvmultiurisrcbin.set_property("live-source", 1)

        # Link elements
        gsthelpers.link_elements([nvmultiurisrcbin, demuxer])

        # Create the image processing pipelines, one for each stream
        for i, el in enumerate(uri_list.split(",")):
            logger.info(f"Connecting processing bin for stream {el}")

            # Create elements
            processing_bin = self.create_processing_bin()

            # Add to the pipeline
            self.pipeline.add(processing_bin)

            # Connect tee to muxer
            src = demuxer.get_request_pad(f"src_{i}")
            if src is None:
                logger.error(
                    f"Failed to request 'src_{i}' pad from the demuxer for stream {el}"
                )
                sys.exit(-1)

            sink = processing_bin.get_static_pad("sink")
            if sink is None:
                logger.error(
                    f"Failed to get 'sink' pad from the processing bin for stream {el}"
                )
                sys.exit(-1)

            if src.link(sink) != Gst.PadLinkReturn.OK:
                logger.error(
                    f"Failed to the demuxer to the processing bin for stream {el}"
                )
                sys.exit(-1)
            else:
                logger.info(
                    f"Linked demuxer 'src_{i}' pad to the processing bin 'sink' pad for stream {el}"
                )

        # Get hold of the bus and add a watcher
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def create_processing_bin(self) -> Gst.Bin:
        """Creates a processor bin

        Returns
        -------
        Gst.Bin
            Created processor bin.
        """

        # Create a bin
        bin = Gst.Bin.new(f"video_processing_bin_{self.bin_cntr}")

        # Create all the elements
        primary_inference = gsthelpers.create_element(
            "nvinferserver", f"primary-inference-{self.bin_cntr}"
        )
        tracker = gsthelpers.create_element("nvtracker", f"tracker-{self.bin_cntr}")
        secondary1_inference = gsthelpers.create_element(
            "nvinferserver", f"secondary1-inference-{self.bin_cntr}"
        )
        secondary2_inference = gsthelpers.create_element(
            "nvinferserver", f"secondary2-inference-{self.bin_cntr}"
        )
        secondary3_inference = gsthelpers.create_element(
            "nvinferserver", f"secondary3-inference-{self.bin_cntr}"
        )
        video_converter = gsthelpers.create_element(
            "nvvideoconvert", f"video-converter-{self.bin_cntr}"
        )
        osd = gsthelpers.create_element("nvdsosd", f"draw-overlays-{self.bin_cntr}")
        videosink_queue = gsthelpers.create_element(
            "queue", f"videosink-queue-{self.bin_cntr}"
        )
        video_sink = gsthelpers.create_element(
            "nveglglessink", f"nvvideo-renderer-{self.bin_cntr}"
        )
        queue1 = gsthelpers.create_element("queue", f"queue1-{self.bin_cntr}")
        queue2 = gsthelpers.create_element("queue", f"queue2-{self.bin_cntr}")
        queue3 = gsthelpers.create_element("queue", f"queue3-{self.bin_cntr}")
        queue4 = gsthelpers.create_element("queue", f"queue4-{self.bin_cntr}")

        # Add elements to the bin
        bin.add(primary_inference)
        bin.add(tracker)
        bin.add(secondary1_inference)
        bin.add(secondary2_inference)
        bin.add(secondary3_inference)
        bin.add(video_converter)
        bin.add(osd)
        bin.add(videosink_queue)
        bin.add(video_sink)
        bin.add(queue1)
        bin.add(queue2)
        bin.add(queue3)
        bin.add(queue4)

        if platform.machine() == "aarch64":
            # Add egl-transform for Jetson
            video_sink_transform = gsthelpers.create_element(
                "nvegltransform", f"video-sink-transform-{self.bin_cntr}"
            )
            bin.add(video_sink_transform)

            # Link inference, tracker and visualization
            gsthelpers.link_elements(
                [
                    queue1,
                    primary_inference,
                    queue2,
                    tracker,
                    queue3,
                    secondary1_inference,
                    secondary2_inference,
                    secondary3_inference,
                    queue4,
                    video_converter,
                    osd,
                    video_sink_transform,
                    video_sink,
                ]
            )
        else:
            # Link inference, tracker and visualization
            gsthelpers.link_elements(
                [
                    queue1,
                    primary_inference,
                    queue2,
                    tracker,
                    queue3,
                    secondary1_inference,
                    secondary2_inference,
                    secondary3_inference,
                    queue4,
                    video_converter,
                    osd,
                    video_sink,
                ]
            )

        # Set properties for the inference engines
        # Since we're reusing the same configuration, we need to tweak the unique-id and infer-on-gie-id
        # in order to avoid clashes
        primary_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt",
        )
        primary_inference.set_property("unique-id", 1 + self.bin_cntr * 10)
        logger.info(
            f"Primary detector unique ID: {primary_inference.get_property('unique-id')}"
        )

        secondary1_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_carcolor.txt",
        )
        secondary1_inference.set_property("unique-id", 2 + self.bin_cntr * 10)
        secondary1_inference.set_property("infer-on-gie-id", 1 + self.bin_cntr * 10)
        logger.info(
            f"Secondary 1 detector unique ID: {secondary1_inference.get_property('unique-id')}"
        )

        secondary2_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_carmake.txt",
        )
        secondary2_inference.set_property("unique-id", 3 + self.bin_cntr * 10)
        secondary2_inference.set_property("infer-on-gie-id", 1 + self.bin_cntr * 10)
        logger.info(
            f"Secondary 2 detector unique ID: {secondary2_inference.get_property('unique-id')}"
        )

        secondary3_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_vehicletypes.txt",
        )
        secondary3_inference.set_property("unique-id", 4 + self.bin_cntr * 10)
        secondary3_inference.set_property("infer-on-gie-id", 1 + self.bin_cntr * 10)
        logger.info(
            f"Secondary 3 detector unique ID: {secondary3_inference.get_property('unique-id')}"
        )

        # Configure tracker
        tracker_config = configparser.ConfigParser()
        tracker_config.read("dstest2_tracker_config.txt")
        for key in tracker_config["tracker"]:
            value = tracker_config["tracker"][key]
            if value.isdigit():
                value = int(value)
            tracker.set_property(key, value)

        # Create ghost pads for external linking
        sink_pad = Gst.GhostPad.new("sink", queue1.get_static_pad("sink"))
        if not sink_pad:
            logger.error("bin failed to create a GhostPad for sink")
            sys.exit(-1)
        bin.add_pad(sink_pad)

        # Add a probe to the sink pad of the osd-element in order to draw/print meta-data to the canvas
        # using the function osd_sink_pad_buffer_probe
        osdsinkpad = osd.get_static_pad("sink")
        assert osdsinkpad is not None
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

        self.bin_cntr += 1

        return bin

    def start(self) -> None:
        logging.info("MultiPlayer starting")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            logging.error("MultiPlayer, failed to start pipeline")
            self.stop()
            return
        else:
            logging.info("MultiPlayer, started")

        # Run the main loop to keep all pipelines running
        self.loop.run()

    def on_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        message_type = message.type

        # End of stream
        if message_type == Gst.MessageType.EOS:
            logging.info("MultiPlayer, EOS received")
            self.stop()

        # Error
        elif message_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error(f"MultiPlayer error {err.message}, debug {debug}")
            self.stop()

        # # State changed
        # elif message_type == Gst.MessageType.STATE_CHANGED:
        #     old_state, new_state, pending = message.parse_state_changed()

        #     src = message.src
        #     if isinstance(src, Gst.Pipeline):
        #         element_name = "Pipeline"
        #     else:
        #         element_name = src.get_name()

        #     logging.info(
        #         f"MultiPlayer, {element_name} state changed: {old_state.value_nick} -> {new_state.value_nick}"
        #     )

    def stop(self):
        logging.info("MultiPlayer stopping")

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()

    def stop_handler(self, sig, frame):
        self.stop()


def main():
    logging.basicConfig(level=logging.INFO)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-u", "--uri", help="Input uri", default="")
    args = argParser.parse_args()

    multi_player = MultiPlayer(args.uri)
    try:
        multi_player.start()
    except Exception as e:
        logging.error(f"Failed to start the pipeline: {e}")
        multi_player.stop()
        sys.exit(-1)

    sys.exit(0)


if __name__ == "__main__":
    main()
