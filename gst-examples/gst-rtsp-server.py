"""
RTSP Server Script

This script creates an RTSP server that streams a media file specified via command-line arguments.
The stream can be played back using an RTSP-compatible player like VLC or ffplay.

Example usage to start the server:
    python3 rtsp_server.py --file /path/to/media/file.mp4

To playback the stream, use one of the following:
    - ffplay rtsp://localhost:8554/test
    - gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/camera1 protocols=tcp latency=500 !
        rtph264depay ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! queue !
        mux.sink_1 nvstreammux name=mux width=1920 height=1080 batch-size=1 live-source=1 ! queue !
        nvvideoconvert ! queue ! nvdsosd ! queue ! nveglglessink

The server will stream the media file over the RTSP protocol, converting raw video to H.264 format.
"""

import gi
import os
import argparse
from gi.repository import Gst, GstRtspServer, GLib

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")


class RTSPServer:
    """RTSP Server to stream media files over RTSP protocol.

    Parameters
    ----------
    file_path : str
        The path to the media file to be streamed.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """

    def __init__(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        Gst.init(None)

        server = GstRtspServer.RTSPServer()
        server.set_address("0.0.0.0")
        server.props.service = "8554"

        factory = GstRtspServer.RTSPMediaFactory()
        launch_str = f"""
            filesrc location="{file_path}" !
            decodebin name=decodebin !
            queue !
            videoconvert !
            x264enc !
            h264parse !
            rtph264pay name=pay0 pt=96 config-interval=1
            """
        factory.set_launch(launch_str)
        factory.set_shared(True)
        factory.set_eos_shutdown(False)  # Ensure the pipeline is created immediately

        # factory.connect("media-configure", self.on_media_configure)

        server.get_mount_points().add_factory("/test", factory)
        server.attach(None)
        print("RTSP server is running on rtsp://0.0.0.0:8554/test")

    def on_media_configure(
        self, factory: GstRtspServer.RTSPMediaFactory, media: GstRtspServer.RTSPMedia
    ) -> None:
        """Configure the media pipeline when it is created.

        Parameters
        ----------
        factory : GstRtspServer.RTSPMediaFactory
            The media factory that triggered this event.
        media : GstRtspServer.RTSPMedia
            The media object that contains the pipeline.
        """
        print("Media is being configured.")
        pipeline = media.get_element()
        bus = pipeline.get_bus()

        bus.add_signal_watch()
        bus.connect("message::state-changed", self.on_state_changed)
        bus.connect("message::error", self.on_error)
        bus.connect("message::eos", self.on_eos)

        decodebin = pipeline.get_by_name("decodebin")
        if decodebin is not None:
            decodebin.connect("pad-added", self.on_pad_added)

    def on_pad_added(self, element: Gst.Element, pad: Gst.Pad) -> None:
        """Handle the addition of a new pad from decodebin.

        Parameters
        ----------
        element : Gst.Element
            The element that generated the pad.
        pad : Gst.Pad
            The pad that was added.
        """
        print(f"New pad '{pad.get_name()}' added.")
        caps = pad.query_caps(None)
        structure_name = caps.get_structure(0).get_name()
        print(f"Pad Caps: {structure_name}")

        if "video" in structure_name:
            print("Video pad detected, linking elements.")
            sink = element.get_static_pad("sink")
            if pad.can_link(sink):
                print(f"Linking pad {pad.get_name()} to sink pad.")
                pad.link(sink)
        else:
            print(
                f"Skipping non-video pad: {pad.get_name()} with caps: {structure_name}"
            )

    def on_state_changed(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle state changes of the GStreamer pipeline.

        Parameters
        ----------
        bus : Gst.Bus
            The bus associated with the pipeline.
        message : Gst.Message
            The message describing the state change.
        """
        if message.src.get_name() == "pipeline0":
            old, new, pending = message.parse_state_changed()
            print(f"Pipeline state changed: {old.value_nick} -> {new.value_nick}")
            if new == Gst.State.PAUSED or new == Gst.State.PLAYING:
                print("Pipeline is ready, attaching bus watch.")

    def on_error(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle error messages from the GStreamer bus.

        Parameters
        ----------
        bus : Gst.Bus
            The bus associated with the pipeline.
        message : Gst.Message
            The message describing the error.
        """
        err, debug = message.parse_error()
        print(f"ERROR: {err}, Debug info: {debug}")

    def on_eos(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle end-of-stream (EOS) messages from the GStreamer bus.

        Parameters
        ----------
        bus : Gst.Bus
            The bus associated with the pipeline.
        message : Gst.Message
            The message indicating that the end of the stream has been reached.
        """
        print("End of stream reached!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RTSP Server to stream a media file over RTSP protocol."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the media file to be streamed."
    )
    args = parser.parse_args()

    file_path: str = args.file
    try:
        server = RTSPServer(file_path)
        loop = GLib.MainLoop()  # Start the GLib main loop
        loop.run()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
