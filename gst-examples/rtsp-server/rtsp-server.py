"""
RTSP Server Script

This script creates an RTSP server that streams a media file specified via command-line arguments.
The stream can be played back using an RTSP-compatible player like VLC or ffplay.

Example usage to start the server:
    python3 rtsp_server.py --file /path/to/media/file.mp4

To playback the stream, use one of the following:
    - ffplay rtsp://localhost:8554/camera1
    - gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/camera1 protocols=tcp latency=500 !
        rtph264depay ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! queue !
        mux.sink_1 nvstreammux name=mux width=1920 height=1080 batch-size=1 live-source=1 ! queue !
        nvvideoconvert ! queue ! nvdsosd ! queue ! nveglglessink

The server will stream the media file over the RTSP protocol, converting raw video to H.264 format.
"""

from urllib.parse import urlparse
import os
import argparse
import logging
from typing import List
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib  # noqa: E402, F401

logger = logging.getLogger(__name__)


class RTSPServer:
    """RTSP Server to stream media files over RTSP protocol using urisrcbin."""

    def __init__(self, uri_list: List[str]) -> None:
        if not uri_list:
            raise ValueError("No input files provided.")

        # If the uri_list contains files, check that these exist
        for uri in uri_list:
            uri_parsed = urlparse(uri)
            if uri_parsed.scheme == "file":
                if not os.path.exists(uri_parsed.path):
                    raise RuntimeError(f"File '{uri_parsed.path}' does not exist")

        Gst.init(None)

        self.server = GstRtspServer.RTSPServer()
        self.server.set_address("0.0.0.0")
        self.server.props.service = "8554"

        self._setup_streams(uri_list)
        self.server.attach(None)
        logging.info("RTSP server is running. Streams available at:")
        for idx, file_path in enumerate(uri_list, 1):
            logging.info(f"{file_path} -> rtsp://0.0.0.0:8554/camera{idx}")

    def _setup_streams(self, file_paths: List[str]) -> None:
        """Set up RTSP streams for each file."""
        mount_points = self.server.get_mount_points()

        for idx, file_path in enumerate(file_paths, 1):
            if not os.path.exists(file_path) and not file_path.startswith("file://"):
                raise FileNotFoundError(f"File or URI not found: {file_path}")

            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_shared(True)
            factory.set_eos_shutdown(False)

            # Use urisrcbin to handle URI inputs
            launch_str = f"""
                urisourcebin uri="{file_path}" !
                queue !
                decodebin name=decodebin !
                queue !
                videoconvert !
                x264enc !
                h264parse !
                rtph264pay name=pay0 pt=96 config-interval=1
            """
            factory.set_launch(launch_str)

            # Add factory to the RTSP server mount point
            mount_points.add_factory(f"/camera{idx}", factory)

    def handle_message_callback(self, message, *user_data):
        logger.info("Message")

    def on_message(bus: Gst.Bus, message: Gst.Message, pipeline: Gst.Pipeline) -> None:
        """
        Handles GStreamer bus messages.

        This inner function listens for End of Stream (EOS) messages and restarts the pipeline
        when EOS is reached.

        Parameters
        ----------
        bus : Gst.Bus
            The GStreamer bus to listen for messages on.
        message : Gst.Message
            The GStreamer message received on the bus.
        pipeline : Gst.Pipeline
            The pipeline to restart if an EOS message is received.

        Returns
        -------
        None
        """
        if message.type == Gst.MessageType.EOS:
            logging.info(
                f"EOS received for pipeline: {pipeline.get_name()}. Restarting pipeline..."
            )
            pipeline.set_state(Gst.State.NULL)  # Stop the pipeline
            pipeline.set_state(Gst.State.PLAYING)  # Restart the pipeline
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logging.error(
                f"Error received from {message.src.get_name()}: {error.message}"
            )
            logging.error(f"Debug info: {debug}")
            pipeline.set_state(Gst.State.NULL)  # Stop on error
        elif message.type == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if message.src == pipeline:
                logging.info(
                    f"Pipeline state changed from {old.value_name} to {new.value_name}."
                )

    def run(self):
        """Start the GLib main loop."""
        try:
            loop = GLib.MainLoop()
            logging.info("Starting GLib Main Loop...")
            loop.run()
        except Exception as e:
            logging.error(f"Error running GLib Main Loop: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RTSP Server to stream multiple media files over RTSP protocol."
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Paths or URIs of the media files to be streamed.",
    )
    args = parser.parse_args()

    file_paths: List[str] = args.files
    try:
        logging.basicConfig(level=logging.INFO)
        server = RTSPServer(file_paths)
        server.run()
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"ERROR: {e}")


if __name__ == "__main__":
    main()
