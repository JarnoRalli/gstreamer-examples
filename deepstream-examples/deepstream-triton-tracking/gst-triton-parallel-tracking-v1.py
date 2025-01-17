"""
This file implements a simple pipeline with detector and classifiers, using Triton Inference Server
for doing inference. The same pipeline can be spawned n-number of times, using the same input. The
idea of this program is to do some testing using Triton.

This example uses a probe attached to the osd plug-in in order to modify the way how the detections
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
>> python3 gst-triton-parallel-tracking-v2.py -h

In order to process a single video file:
>> python3 gst-triton-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

In order to process the same input file in parallel:
>> python3 gst-triton-tracking-v2.py -n 4 -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
"""

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

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

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


def osd_sink_pad_buffer_probe(pad, info, u_data):
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


class Player(object):
    """
    A simple Player-class that processes files with h264 encoded video content.
    """

    def __init__(self):

        # Initialize gst
        Gst.init(None)

        # Create mainloop
        self.loop = GLib.MainLoop()

        # Register a signal handler for SIGINT
        signal.signal(signal.SIGINT, self.stop_handler)
        signal.signal(signal.SIGTERM, self.stop_handler)
        signal.signal(signal.SIGHUP, self.stop_handler)

        # Create an empty pipeline
        self.pipeline = Gst.Pipeline.new("video-pipeline")
        assert self.pipeline is not None

        # Create all the elements
        self.source = gsthelpers.create_element("filesrc", "source")
        self.demuxer = gsthelpers.create_element("qtdemux", "demuxer")
        self.video_queue = gsthelpers.create_element("queue", "video-queue")
        self.h264_parser = gsthelpers.create_element("h264parse", "h264-parser")
        self.h264_decoder = gsthelpers.create_element("nvv4l2decoder", "h264-decoder")
        self.stream_muxer = gsthelpers.create_element("nvstreammux", "stream-muxer")
        self.primary_inference = gsthelpers.create_element(
            "nvinferserver", "primary-inference"
        )
        self.tracker = gsthelpers.create_element("nvtracker", "tracker")
        self.secondary1_inference = gsthelpers.create_element(
            "nvinferserver", "secondary1-inference"
        )
        self.secondary2_inference = gsthelpers.create_element(
            "nvinferserver", "secondary2-inference"
        )
        self.secondary3_inference = gsthelpers.create_element(
            "nvinferserver", "secondary3-inference"
        )
        self.video_converter = gsthelpers.create_element(
            "nvvideoconvert", "video-converter"
        )
        self.osd = gsthelpers.create_element("nvdsosd", "nvidia-bounding-box-draw")
        self.tee = gsthelpers.create_element("tee", "tee")
        # Video sink branch
        self.videosink_queue = gsthelpers.create_element("queue", "videosink-queue")
        self.video_sink = gsthelpers.create_element("nveglglessink", "nvvideo-renderer")

        # Add elements to the pipeline
        self.pipeline.add(self.source)
        self.pipeline.add(self.demuxer)
        self.pipeline.add(self.video_queue)
        self.pipeline.add(self.h264_parser)
        self.pipeline.add(self.h264_decoder)
        self.pipeline.add(self.stream_muxer)
        self.pipeline.add(self.primary_inference)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.secondary1_inference)
        self.pipeline.add(self.secondary2_inference)
        self.pipeline.add(self.secondary3_inference)
        self.pipeline.add(self.video_converter)
        self.pipeline.add(self.osd)
        self.pipeline.add(self.tee)
        # Video sink branch
        self.pipeline.add(self.videosink_queue)
        self.pipeline.add(self.video_sink)

        # Set properties for the streammux
        self.stream_muxer.set_property("width", 1920)
        self.stream_muxer.set_property("height", 1080)
        self.stream_muxer.set_property("batch-size", 1)
        self.stream_muxer.set_property("batched-push-timeout", 4000000)

        # Set properties for sinks
        self.video_sink.set_property("async", False)

        # Set properties for the inference engines
        self.primary_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt",
        )
        self.secondary1_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_carcolor.txt",
        )
        self.secondary2_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_carmake.txt",
        )
        self.secondary3_inference.set_property(
            "config-file-path",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_vehicletypes.txt",
        )

        # Set properties for the tracker
        tracker_config = configparser.ConfigParser()
        tracker_config.read("dstest2_tracker_config.txt")
        tracker_config.sections()

        for key in tracker_config["tracker"]:
            if key == "tracker-width":
                tracker_width = tracker_config.getint("tracker", key)
                self.tracker.set_property("tracker-width", tracker_width)
            if key == "tracker-height":
                tracker_height = tracker_config.getint("tracker", key)
                self.tracker.set_property("tracker-height", tracker_height)
            if key == "gpu-id":
                tracker_gpu_id = tracker_config.getint("tracker", key)
                self.tracker.set_property("gpu_id", tracker_gpu_id)
            if key == "ll-lib-file":
                tracker_ll_lib_file = tracker_config.get("tracker", key)
                self.tracker.set_property("ll-lib-file", tracker_ll_lib_file)
            if key == "ll-config-file":
                tracker_ll_config_file = tracker_config.get("tracker", key)
                self.tracker.set_property("ll-config-file", tracker_ll_config_file)
            if key == "enable-batch-process":
                tracker_enable_batch_process = tracker_config.getint("tracker", key)
                self.tracker.set_property(
                    "enable_batch_process", tracker_enable_batch_process
                )
            if key == "enable-past-frame":
                tracker_enable_past_frame = tracker_config.getint("tracker", key)
                self.tracker.set_property(
                    "enable_past_frame", tracker_enable_past_frame
                )

        # --- LINK IMAGE PROCESSING ---
        # Link video input and inference as follows:
        #
        # filesrc -> demux -> queue -> h264parser -> h264decoder -> streammux ->
        # primary_inference1 -> tracker -> secondary_inference1 -> secondary_inference2 -> secondary_inference3 ->
        # videoconverter -> osd (bounding boxes) -> tee
        #
        # After the tee element we have two output branches that are described later.

        # Link source to demuxer
        gsthelpers.link_elements([self.source, self.demuxer])

        # Connect demux to the pad-added signal, used to link demuxer to queue dynamically
        demuxer_pad_added = gsthelpers.PadAddedLinkFunctor()
        demuxer_pad_added.register("video_", self.video_queue, "sink")

        assert self.demuxer.connect("pad-added", demuxer_pad_added) is not None

        # Link video pipeline
        gsthelpers.link_elements(
            [self.video_queue, self.h264_parser, self.h264_decoder]
        )

        # Link decoder to streammux
        source = self.h264_decoder.get_static_pad("src")
        assert source is not None
        sink = self.stream_muxer.get_request_pad("sink_0")
        assert sink is not None
        assert source.link(sink) == Gst.PadLinkReturn.OK

        # Link inference, tracker and visualization
        gsthelpers.link_elements(
            [
                self.stream_muxer,
                self.primary_inference,
                self.tracker,
                self.secondary1_inference,
                self.secondary2_inference,
                self.secondary3_inference,
                self.video_converter,
                self.osd,
                self.tee,
            ]
        )

        # --- LINK OUTPUT BRANCHE ---
        # We have videosink output as follows:
        #
        # osd -> tee -> queue -> videosink
        #

        # --- Video-sink output branch ---
        src = self.tee.get_request_pad("src_0")
        assert src is not None
        sink = self.videosink_queue.get_static_pad("sink")
        assert sink is not None
        assert src.link(sink) == Gst.PadLinkReturn.OK

        # Link video_queue to video_sink
        gsthelpers.link_elements([self.videosink_queue, self.video_sink])

        # --- Meta-data output ---
        # Add a probe to the sink pad of the osd-element in order to draw/print meta-data to the canvas
        osdsinkpad = self.osd.get_static_pad("sink")
        assert osdsinkpad is not None
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    def play(self, input_file: str):
        """

        :param input_file: path to the h264 encoded input file
        :return:
        """

        print(f"PLAY(input_file={input_file})")

        # Check if the file exists
        if not os.path.exists(input_file):
            raise RuntimeError(f"Input file '{input_file}' does not exist")

        # Set source location property to the file location
        self.source.set_property("location", input_file)

        # Create a bus and add signal watcher
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        print("Setting pipeline state to PLAYING...", end="")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print("failed")
        else:
            print("done")

        # Start loop
        self.loop.run()

    def stop(self):
        print("STOP()")
        print("Setting pipeline state to NULL...", end="")
        self.pipeline.set_state(Gst.State.NULL)
        print("done")
        self.loop.quit()

    def on_message(self, bus, message):
        """
        Message handler function.

        :param bus: bus
        :param message: message
        :return: nothing
        """
        message_type = message.type
        if message_type == Gst.MessageType.EOS:
            print("EOS message type received")
            self.stop()

        elif message_type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"Error from {message.src.get_name()}: {err.message}")
            self.stop()

    def stop_handler(self, sig, frame):
        """
        Even handler for stopping the pipeline.

        :param sig: signal
        :param frame: stack frame
        :return:
        """
        print("Signal SIGINT received")
        self.stop()


class MultiPipelinePlayer:
    def __init__(self, num_pipelines, input_file):
        Gst.init(None)
        self.loop = GLib.MainLoop()
        signal.signal(signal.SIGINT, self.stop_handler)
        self.pipelines = []
        self.num_pipelines = num_pipelines

        for i in range(num_pipelines):
            # Create a Player instance for each pipeline
            player = Player()

            # Add player and its corresponding output file to the list
            self.pipelines.append(player)

    def start(self, input_file):
        # Configure each pipeline with the same input file but different output files
        for i, player in enumerate(self.pipelines):
            print(f"Starting pipeline {i+1}")
            player.source.set_property("location", input_file)

            # Create a bus for each pipeline
            bus = player.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message, player)

            # Set the pipeline to PLAYING state
            player.pipeline.set_state(Gst.State.PLAYING)

        # Run the main loop to keep all pipelines running
        self.loop.run()

    def on_message(self, bus, message, player):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print("End of stream reached for a pipeline.")
            player.stop()
            self.check_all_pipelines_stopped()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            player.stop()
            self.check_all_pipelines_stopped()

    def check_all_pipelines_stopped(self):
        if all(
            player.pipeline.get_state(Gst.CLOCK_TIME_NONE)[1] == Gst.State.NULL
            for player in self.pipelines
        ):
            print("All pipelines have stopped. Exiting...")
            self.loop.quit()

    def stop(self):
        print("Stopping all pipelines...")
        for player in self.pipelines:
            player.stop()
        self.loop.quit()

    def stop_handler(self, sig, frame):
        self.stop()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_file", help="input file path", default="")
    argParser.add_argument(
        "-n", "--num_pipelines", help="Number of pipelines to run", type=int, default=1
    )
    args = argParser.parse_args()

    multi_player = MultiPipelinePlayer(args.num_pipelines, args.input_file)
    try:
        multi_player.start(args.input_file)
    except Exception as e:
        print(e)
        multi_player.stop()
        sys.exit(-1)

    sys.exit(0)
