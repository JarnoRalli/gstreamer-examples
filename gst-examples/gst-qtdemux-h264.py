"""
This example shows how to dynamically link a demux-element. Demux-elements don't have static pads, instead
the pads are created dynamically based on the contents of the stream. For example, if the stream
doesn't contain audio, then a pad related to an audio stream is not created.

This example plays back a video file, encoded using h264, from a container like mp4.

For help regarding command line arguments, execute the following:

python3 gst-qtdemux-h264.py -h
"""

import argparse
import os
import sys
import signal
from helpers import *

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject


class Player(object):
    """
    A simple Player-class that plays back files with h264 encoded video content.
    Can play, for example, mp4 files that have h264 encoded video.
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
        # File source
        self.source = gsthelpers.create_element("filesrc", "source")

        # Demuxer
        self.demuxer = gsthelpers.create_element("qtdemux", "demuxer")

        # Video pipeline
        self.video_queue = gsthelpers.create_element("queue", "video-queue")
        self.h264_parser = gsthelpers.create_element("h264parse", "h264-parser")
        self.h264_decoder = gsthelpers.create_element("avdec_h264", "h264-decoder")
        self.video_converter = gsthelpers.create_element("videoconvert", "video-converter")
        self.image_sink = gsthelpers.create_element("xvimagesink", "image-sink")

        # Add elements to the pipeline
        self.pipeline.add(self.source)
        self.pipeline.add(self.demuxer)
        self.pipeline.add(self.video_queue)
        self.pipeline.add(self.h264_parser)
        self.pipeline.add(self.h264_decoder)
        self.pipeline.add(self.video_converter)
        self.pipeline.add(self.image_sink)

        # Link source to demuxer
        gsthelpers.link_elements([self.source, self.demuxer])

        # Link video pipeline
        gsthelpers.link_elements([self.video_queue,
                                  self.h264_parser,
                                  self.h264_decoder,
                                  self.video_converter,
                                  self.image_sink])

        # Connect demux to the pad-added signal, used to link queue to parser dynamically
        pad_added_functor = gsthelpers.PadAddedLinkFunctor()
        pad_added_functor.register("video_", self.video_queue, "sink")

        assert self.demuxer.connect("pad-added", pad_added_functor)

    def play(self, input_file: str) -> None:
        """
        Plays back a file.

        :param input_file: path to the file to be played back
        :return: nothing
        """

        print(f"PLAY(input_file={input_file})")

        # Check if the file exists
        if not os.path.exists(input_file):
            raise RuntimeError(f"File '{input_file}' does not exist")

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
        print("STOP")
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


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_file", help="path to the file to be played back")
    args = argParser.parse_args()

    player = Player()
    try:
        player.play(args.input_file)
    except Exception as e:
        print(e)
        player.stop()
        sys.exit(-1)

    sys.exit(0)
