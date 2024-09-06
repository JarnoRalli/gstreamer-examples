"""
This file re-implements deepstream (Python) example deepstream-test2.py, using Triton Inference Server.
In essence the example reads a h264 encoded video stream from a file, like mp4, and tracks objects like:

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

For more information regarding the input parameters, execute the following:

python3 gst-triton-tracking.py -h

In order to process a file:

python3 gst-triton-tracking.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
"""

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


# This function is copied from:
# https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-test2/deepstream_test_2.py
def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    # Initialising object counter with 0.
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

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(
            frame_number,
            num_rects,
            obj_counter[PGIE_CLASS_ID_VEHICLE],
            obj_counter[PGIE_CLASS_ID_PERSON],
        )

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # past tracking meta data
    if past_tracking_meta[0] == 1:
        l_user = batch_meta.batch_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if (
                user_meta
                and user_meta.base_meta.meta_type
                == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META
            ):
                try:
                    # Note that user_meta.user_meta_data needs a cast to pyds.NvDsPastFrameObjBatch
                    # The casting is done by pyds.NvDsPastFrameObjBatch.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone
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
        self.decoder_queue = gsthelpers.create_element("queue", "decoder-queue")
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
        # File sink branch
        self.filesink_queue = gsthelpers.create_element("queue", "filesink-queue")
        self.file_sink_converter = gsthelpers.create_element(
            "nvvideoconvert", "file-sink-videoconverter"
        )
        self.file_sink_encoder = gsthelpers.create_element(
            "x264enc", "file-sink-encoder"
        )
        self.file_sink_parser = gsthelpers.create_element(
            "h264parse", "file-sink-parser"
        )
        self.file_sink_muxer = gsthelpers.create_element(
            "matroskamux", "file-sink-muxer"
        )
        self.file_sink = gsthelpers.create_element("filesink", "file-sink")

        # Add elements to the pipeline
        self.pipeline.add(self.source)
        self.pipeline.add(self.demuxer)
        self.pipeline.add(self.video_queue)
        self.pipeline.add(self.h264_parser)
        self.pipeline.add(self.h264_decoder)
        self.pipeline.add(self.decoder_queue)
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
        # File sink branch
        self.pipeline.add(self.filesink_queue)
        self.pipeline.add(self.file_sink_converter)
        self.pipeline.add(self.file_sink_encoder)
        self.pipeline.add(self.file_sink_parser)
        self.pipeline.add(self.file_sink_muxer)
        self.pipeline.add(self.file_sink)

        # Set properties for the streammux
        self.stream_muxer.set_property("width", 1920)
        self.stream_muxer.set_property("height", 1080)
        self.stream_muxer.set_property("batch-size", 1)
        self.stream_muxer.set_property("batched-push-timeout", 4000000)

        # Set properties for sinks
        self.video_sink.set_property("async", False)
        self.file_sink.set_property("sync", False)
        self.file_sink.set_property("async", False)

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
            [self.video_queue, self.h264_parser, self.h264_decoder, self.decoder_queue]
        )

        # Link decoder to streammux
        source = self.decoder_queue.get_static_pad("src")
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

        # Link video_queue to video_sink
        gsthelpers.link_elements([self.videosink_queue, self.video_sink])

        # --- File-sink output branch ---
        src = self.tee.get_request_pad("src_1")
        assert src is not None
        sink = self.filesink_queue.get_static_pad("sink")
        assert sink is not None
        assert src.link(sink) == Gst.PadLinkReturn.OK

        gsthelpers.link_elements(
            [
                self.filesink_queue,
                self.file_sink_converter,
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

    def play(self, input_file: str, output_file: str):
        """

        :param input_file: path to the h264 encoded input file
        :param output_file: path to the h264 encoded output file
        :return:
        """

        print(f"PLAY(input_file={input_file}, output_file={output_file})")

        # Check if the file exists
        if not os.path.exists(input_file):
            raise RuntimeError(f"Input file '{input_file}' does not exist")

        # Set source location property to the file location
        self.source.set_property("location", input_file)

        # Set location for the output file
        self.file_sink.set_property("location", output_file)

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

        elif message_type == Gst.MessageType.WARNING:
            err, dbg = message.parse_error()
            print(f"Warning from {message.src.get_name()}: {err.message}")

        elif message_type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            print(
                f"State changed from {message.src.get_name()}: {old_state.value_nick} -> {new_state.value_nick}, pending: {pending_state.value_nick}"
            )

    def stop_handler(self, sig, frame):
        """
        Even handler for stopping the pipeline.

        :param sig: signal
        :param frame: stack frame
        :return:
        """
        print("Signal SIGINT received")
        self.stop()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_file", help="input file path", default="")
    argParser.add_argument(
        "-o", "--output_file", help="output file path", default="output.mp4"
    )
    args = argParser.parse_args()

    player = Player()
    try:
        player.play(args.input_file, args.output_file)
    except Exception as e:
        print(e)
        player.stop()
        sys.exit(-1)

    sys.exit(0)
