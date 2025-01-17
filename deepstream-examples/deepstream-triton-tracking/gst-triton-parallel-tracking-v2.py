"""
This file implements a parallel pipeline, using Triton server, that allows to process several
h264 encoded video files in parallel.

In order to get help regarding input parameters:
>> python3 gst-triton-parallel-tracking-v2.py -h

In order to process a video file:
>> python3 gst-triton-parallel-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

In order to process 2 video files:
>> python3 gst-triton-parallel-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \\
>>   /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
"""

import argparse
import configparser
import sys
import signal
from helpers import gsthelpers
import gi
import math

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


def set_tiler_layout(tiler, num_streams, tile_width=1920, tile_height=1080):
    """
    Automatically calculate rows and columns for the tiler and set its properties.

    :param tiler: The nvmultistreamtiler element.
    :param num_streams: Number of input streams.
    :param tile_width: Width of the tiled output.
    :param tile_height: Height of the tiled output.
    """
    # Calculate number of rows and columns
    rows = math.ceil(math.sqrt(num_streams))
    columns = math.ceil(num_streams / rows)

    # Set properties for the tiler
    tiler.set_property("rows", rows)
    tiler.set_property("columns", columns)
    tiler.set_property("width", tile_width)
    tiler.set_property("height", tile_height)

    print(
        f"Tiler layout: {rows} rows x {columns} columns, Output size: {tile_width}x{tile_height}"
    )


class Player:
    def __init__(self, input_files):

        Gst.init(None)
        self.loop = GLib.MainLoop()
        signal.signal(signal.SIGINT, self.stop_handler)

        # Register signal handlers
        signal.signal(signal.SIGINT, self.stop_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, self.stop_handler)  # Handle termination signals

        # Create pipeline
        self.pipeline = Gst.Pipeline.new("multi-stream-pipeline")

        # Elements
        self.stream_muxer = gsthelpers.create_element("nvstreammux", "stream-muxer")
        self.primary_inference = gsthelpers.create_element(
            "nvinferserver", "primary-inference"
        )
        self.secondary1_inference = gsthelpers.create_element(
            "nvinferserver", "secondary1-inference"
        )
        self.secondary2_inference = gsthelpers.create_element(
            "nvinferserver", "secondary2-inference"
        )
        self.secondary3_inference = gsthelpers.create_element(
            "nvinferserver", "secondary3-inference"
        )
        self.tracker = gsthelpers.create_element("nvtracker", "tracker")
        self.tiler = gsthelpers.create_element("nvmultistreamtiler", "tiler")
        self.video_converter = gsthelpers.create_element(
            "nvvideoconvert", "video-converter"
        )
        self.osd = gsthelpers.create_element("nvdsosd", "nvidia-bounding-box-draw")
        self.video_sink = gsthelpers.create_element("nveglglessink", "video-sink")

        # Configure streammux
        self.stream_muxer.set_property("width", 1920)
        self.stream_muxer.set_property("height", 1080)
        self.stream_muxer.set_property("batch-size", len(input_files))
        self.stream_muxer.set_property("batched-push-timeout", 4000000)
        self.stream_muxer.set_property("attach-sys-ts", True)
        self.stream_muxer.set_property("enable-padding", True)

        # Configure video sink
        self.video_sink.set_property("sync", True)

        # Configure inference engines
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

        # Configure tracker
        tracker_config = configparser.ConfigParser()
        tracker_config.read("dstest2_tracker_config.txt")
        for key in tracker_config["tracker"]:
            value = tracker_config["tracker"][key]
            if value.isdigit():
                value = int(value)
            self.tracker.set_property(key, value)

        # Configure tiler
        set_tiler_layout(self.tiler, len(input_files))

        # Add elements to pipeline
        self.pipeline.add(self.stream_muxer)
        self.pipeline.add(self.primary_inference)
        self.pipeline.add(self.secondary1_inference)
        self.pipeline.add(self.secondary2_inference)
        self.pipeline.add(self.secondary3_inference)
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.tiler)
        self.pipeline.add(self.video_converter)
        self.pipeline.add(self.osd)
        self.pipeline.add(self.video_sink)

        # Link elements
        gsthelpers.link_elements(
            [
                self.stream_muxer,
                self.primary_inference,
                self.secondary1_inference,
                self.secondary2_inference,
                self.secondary3_inference,
                self.tracker,
                self.tiler,
                self.video_converter,
                self.osd,
                self.video_sink,
            ]
        )

        # Add sources dynamically
        for i, input_file in enumerate(input_files):
            # Create source elements
            source = gsthelpers.create_element("filesrc", f"source-{i}")
            source.set_property("location", input_file)
            demuxer = gsthelpers.create_element("qtdemux", f"demuxer-{i}")
            parser = gsthelpers.create_element("h264parse", f"parser-{i}")
            decoder = gsthelpers.create_element("nvv4l2decoder", f"decoder-{i}")

            # Add elements to the pipeline
            self.pipeline.add(source)
            self.pipeline.add(demuxer)
            self.pipeline.add(parser)
            self.pipeline.add(decoder)

            # Link source to demuxer
            gsthelpers.link_elements([source, demuxer])

            # Connect to pad-added signal of the demuxer
            def on_pad_added(demuxer, pad, parser):
                sink_pad = parser.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)

            demuxer.connect("pad-added", on_pad_added, parser)

            # Link parser to decoder
            gsthelpers.link_elements([parser, decoder])

            # Connect decoder to streammux
            sink_pad = self.stream_muxer.get_request_pad(f"sink_{i}")
            src_pad = decoder.get_static_pad("src")
            src_pad.link(sink_pad)

    def play(self):

        # Create a bus and add signal watcher
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        print("Setting pipeline state to PLAYING...")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to PLAYING")
        else:
            print("Pipeline is now PLAYING")
        self.loop.run()

    def on_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print("All streams have sent EOS. Stopping pipeline...")
            self.stop()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error from {message.src.get_name()}: {err.message}")
            self.stop()

    def stop(self):
        print("Stopping pipeline...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)  # Transition to NULL state
        self.loop.quit()  # Quit the GLib main loop
        print("Pipeline stopped.")

    def stop_handler(self, sig, frame):
        print("Signal received. Stopping pipeline...")
        self.stop()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-i", "--input_files", nargs="+", help="Input video files", required=True
    )
    args = argParser.parse_args()

    player = Player(args.input_files)
    try:
        player.play()
    except Exception as e:
        print(f"Error: {e}")
        player.stop()
        sys.exit(-1)

    sys.exit(0)
