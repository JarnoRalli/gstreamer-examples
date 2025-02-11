"""
This file re-implements deepstream (Python) example deepstream-test2.py, hopefully in a cleaner manner. In essence
the example reads a h264 encoded video stream from a file, like mp4, and tracks objects like:

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

This version draws information of objects that are further away from the camera first.
Everything else being same, due to projective geometry, objects that are smaller are further
away from the camera. By drawing bounding boxes and labels for objects that are further
away from the camera first, object IDs and labels will be easier to read.

For more information regarding the input parameters, execute the following:

python3 gst-tracking-v2.py -h

In order to process a file:

python3 gst-tracking-v2.py -u file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
"""

from collections import namedtuple
from operator import attrgetter
import argparse
import configparser
import sys
import signal
import pyds
from helpers import gsthelpers
import gi
import logging
import platform

logger = logging.getLogger(__name__)

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
    A simple Player-class that processes streams based on a give URI
    """

    def __init__(self, output_file: str = ""):

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
        self.urisrcbin = gsthelpers.create_element("nvurisrcbin", "urisrcbin")
        self.video_queue = gsthelpers.create_element("queue", "video-queue")
        self.stream_muxer = gsthelpers.create_element("nvstreammux", "stream-muxer")
        self.primary_inference = gsthelpers.create_element(
            "nvinfer", "primary-inference"
        )
        self.tracker = gsthelpers.create_element("nvtracker", "tracker")
        self.secondary1_inference = gsthelpers.create_element(
            "nvinfer", "secondary1-inference"
        )
        self.secondary2_inference = gsthelpers.create_element(
            "nvinfer", "secondary2-inference"
        )
        self.secondary3_inference = gsthelpers.create_element(
            "nvinfer", "secondary3-inference"
        )
        self.video_converter = gsthelpers.create_element(
            "nvvideoconvert", "video-converter"
        )
        self.osd = gsthelpers.create_element("nvdsosd", "nvidia-bounding-box-draw")
        self.tee = gsthelpers.create_element("tee", "tee")
        # Video sink branch
        self.videosink_queue = gsthelpers.create_element("queue", "videosink-queue")
        self.video_sink = gsthelpers.create_element("nveglglessink", "nvvideo-renderer")
        # File sink branch
        if output_file != "":
            self.filesink_queue = gsthelpers.create_element("queue", "filesink-queue")
            self.file_sink_converter = gsthelpers.create_element(
                "nvvideoconvert", "file-sink-videoconverter"
            )
            self.caps_filter = gsthelpers.create_element("capsfilter", "capsfilter")
            self.file_sink_encoder = gsthelpers.create_element(
                "nvv4l2h264enc", "file-sink-encoder"
            )
            self.file_sink_parser = gsthelpers.create_element(
                "h264parse", "file-sink-parser"
            )
            self.file_sink_muxer = gsthelpers.create_element(
                "matroskamux", "file-sink-muxer"
            )
            self.file_sink = gsthelpers.create_element("filesink", "file-sink")

        # Add elements to the pipeline
        self.pipeline.add(self.urisrcbin)
        self.pipeline.add(self.stream_muxer)
        self.pipeline.add(self.video_queue)
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
        # File sink branch
        if output_file != "":
            self.pipeline.add(self.filesink_queue)
            self.pipeline.add(self.file_sink_converter)
            self.pipeline.add(self.caps_filter)
            self.pipeline.add(self.file_sink_encoder)
            self.pipeline.add(self.file_sink_parser)
            self.pipeline.add(self.file_sink_muxer)
            self.pipeline.add(self.file_sink)
            # Set properties for file_sink_encoder
            self.file_sink_encoder.set_property("profile", 4)
            # Set properties for the caps filter
            self.caps_filter.set_property(
                "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
            )

        # Set properties for the nvusrsrcbin
        # self.urisrcbin.set_property("cudadec-memtype", 2)

        # Set properties for the streammux
        self.stream_muxer.set_property("width", 1920)
        self.stream_muxer.set_property("height", 1080)
        self.stream_muxer.set_property("batch-size", 1)
        self.stream_muxer.set_property("batched-push-timeout", 4000000)
        self.stream_muxer.set_property("attach-sys-ts", True)
        self.stream_muxer.set_property("enable-padding", True)
        # self.stream_muxer.set_property("live-source", 1)

        # Set properties for the inference engines
        self.primary_inference.set_property(
            "config-file-path", "dstest2_pgie_config.txt"
        )
        self.secondary1_inference.set_property(
            "config-file-path", "dstest2_sgie1_config.txt"
        )
        self.secondary2_inference.set_property(
            "config-file-path", "dstest2_sgie2_config.txt"
        )
        self.secondary3_inference.set_property(
            "config-file-path", "dstest2_sgie3_config.txt"
        )

        # Configure tracker
        tracker_config = configparser.ConfigParser()
        tracker_config.read("dstest2_tracker_config.txt")
        for key in tracker_config["tracker"]:
            value = tracker_config["tracker"][key]
            if value.isdigit():
                value = int(value)
            self.tracker.set_property(key, value)

        # Set video sink properties
        self.video_sink.set_property("sync", 1)

        # --- LINK IMAGE PROCESSING ---
        # Link video input and inference as follows:
        #
        # urisrcbin -> streammux -> video_queue -> primary_inference1 -> tracker
        # -> secondary_inference1 -> secondary_inference2 -> secondary_inference3
        # -> videoconverter -> osd (bounding boxes) -> tee
        #
        # After the tee element we have two output branches that are described later.

        # Connect urisrcbin to the pad-added signal, used for linking urisrcbin to streammux dynamically
        self.urisrcbin.connect("pad-added", self.on_pad_added, "vsrc")

        # Link inference, tracker and visualization
        gsthelpers.link_elements(
            [
                self.stream_muxer,
                self.video_queue,
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

        # --- LINK OUTPUT BRANCHES ---
        # We have two outputs, videosink and a filesink, as follows:
        #
        #             |-> queue -> videosink
        # osd -> tee -|
        #             |-> queue -> videoconvert -> h264enc -> h264parse -> matroskamux -> filesink

        # --- Video-sink output branch ---
        src = self.tee.get_request_pad("src_0")
        assert src is not None
        sink = self.videosink_queue.get_static_pad("sink")
        assert sink is not None
        assert src.link(sink) == Gst.PadLinkReturn.OK

        # If Jetson
        if platform.machine() == "aarch64":
            self.video_sink_transform = gsthelpers.create_element(
                "nvegltransform", "video-sink-transform"
            )
            self.pipeline.add(self.video_sink_transform)
            gsthelpers.link_elements(
                [self.videosink_queue, self.video_sink_transform, self.video_sink]
            )
        # Non-jetson
        else:
            gsthelpers.link_elements([self.videosink_queue, self.video_sink])

        # --- File-sink output branch ---
        if output_file != "":
            src = self.tee.get_request_pad("src_1")
            assert src is not None
            sink = self.filesink_queue.get_static_pad("sink")
            assert sink is not None
            assert src.link(sink) == Gst.PadLinkReturn.OK

            gsthelpers.link_elements(
                [
                    self.filesink_queue,
                    self.file_sink_converter,
                    self.caps_filter,
                    self.file_sink_encoder,
                    self.file_sink_parser,
                ]
            )

            src = self.file_sink_parser.get_static_pad("src")
            assert src is not None
            sink = self.file_sink_muxer.get_request_pad("video_0")
            assert sink is not None
            assert src.link(sink) == Gst.PadLinkReturn.OK

            gsthelpers.link_elements([self.file_sink_muxer, self.file_sink])

        # --- Meta-data output ---
        # Add a probe to the sink pad of the osd-element in order to draw/print meta-data to the canvas
        osdsinkpad = self.osd.get_static_pad("sink")
        assert osdsinkpad is not None
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    def on_pad_added(self, src, new_pad, user_data: str):

        logger.info(f"Received new pad '{new_pad.get_name()}' from '{src.get_name()}'")

        # Check that the new_pad name starts with the name given by the user
        if new_pad.get_name().startswith(user_data):

            # Request a sink pad from the streammuxer
            sink_pad = self.stream_muxer.get_request_pad("sink_0")

            if not sink_pad:
                raise RuntimeError("Could not get a sink pad from the streammuxer")

            # Link the pad
            if new_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Failed to link {new_pad.get_name()} to {sink_pad.get_name()}"
                )
            else:
                logger.info(
                    f"Connected '{new_pad.get_name()}' to '{sink_pad.get_name()}'"
                )

    def play(self, uri: str, output_file: str):
        """

        :param uri: URI of the file or rtsp source
        :param output_file: path to the h264 encoded output file
        :return:
        """

        logger.info(f"PLAY(uri={uri}, output_file={output_file})")

        # Set source location property to the file location
        self.urisrcbin.set_property("uri", uri)

        # # Set location for the output file
        # self.file_sink.set_property("location", output_file)

        # Create a bus and add signal watcher
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        logger.info("Setting pipeline state to PLAYING")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to set the pipeline to state PLAYING")
        else:
            logger.info("Pipeline set to state PLAYING")

        # Start loop
        self.loop.run()

    def stop(self):
        logger.info("Stopping the pipeline")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
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
            logger.info("EOS message type received")
            self.stop()

        elif message_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.info(f"Error from {message.src.get_name()}: {err.message}, {debug}")
            self.stop()

        # State changed
        elif message_type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending = message.parse_state_changed()

            src = message.src
            if isinstance(src, Gst.Pipeline):
                element_name = "Player"
            else:
                element_name = src.get_name()

            logging.info(
                f"Player, {element_name} state changed: {old_state.value_nick} -> {new_state.value_nick}"
            )

    def stop_handler(self, sig, frame):
        """
        Even handler for stopping the pipeline.

        :param sig: signal
        :param frame: stack frame
        :return:
        """
        logger.info("Signal SIGINT received")
        self.stop()


def main():
    logging.basicConfig(level=logging.INFO)
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-u", "--uri", help="URI of the file or rtsp source", default=""
    )
    argParser.add_argument("-o", "--output_file", help="output file path", default="")
    args = argParser.parse_args()

    player = Player(output_file=args.output_file)
    try:
        player.play(args.uri, args.output_file)
    except Exception as e:
        print(e)
        player.stop()
        sys.exit(-1)

    sys.exit(0)


if __name__ == "__main__":
    main()
