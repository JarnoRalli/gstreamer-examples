#!/usr/bin/env python3
"""
GStreamer Python plugin example implementing object tracking after yoloxtensordec.

This script registers a custom GStreamer element called 'gstbytetrack' in-memory,
sets up a video processing pipeline, and runs the main loop to process
the input video file.
"""

import argparse
import sys
import os
from typing import List, Tuple, Dict, Any, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstAnalytics", "1.0")
from gi.repository import Gst, GstBase, GstAnalytics, GLib  # noqa: E402

# Initialize GStreamer before defining any Gst-derived classes
Gst.init(None)


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    boxA : List[float]
        Bounding box in format [x, y, w, h].
    boxB : List[float]
        Bounding box in format [x, y, w, h].

    Returns
    -------
    float
        The intersection-over-union ratio, between 0.0 and 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0.0:
        return 0.0
    return interArea / unionArea


class SimpleTracker:
    """
    A simple Intersection-over-Union (IoU) tracker for bounding boxes.

    This class serves as a functional placeholder for ByteTrack, tracking
    objects between frames by matching bounding boxes with the highest IoU.

    Attributes
    ----------
    next_id : int
        The next unique integer ID to assign to a tracked object.
    tracks : dict
        A dictionary mapping object IDs to their last known bounding box
        represented as a list or tuple: [x, y, w, h].
    """

    def __init__(self) -> None:
        """Initialize the tracker with empty tracks and ID counter."""
        self.next_id: int = 1
        self.tracks: Dict[int, List[float]] = {}

    def update(self, detections: List[List[float]]) -> List[Tuple[int, List[float]]]:
        """
        Update the tracker with new detections from the current frame.

        Parameters
        ----------
        detections : List[List[float]]
            A list of bounding boxes detected in the current frame,
            where each box is [x, y, w, h].

        Returns
        -------
        List[Tuple[int, List[float]]]
            A list of tuples, each containing the tracking ID and its corresponding
            updated bounding box.
        """
        new_tracks: Dict[int, List[float]] = {}
        matched_detections = set()

        # Try to match existing tracks with new detections
        for track_id, last_box in self.tracks.items():
            best_iou = 0.3  # Threshold
            best_idx = -1
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                iou = compute_iou(last_box, det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx != -1:
                new_tracks[track_id] = detections[best_idx]
                matched_detections.add(best_idx)

        # Assign new IDs to remaining unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                new_tracks[self.next_id] = det
                self.next_id += 1

        self.tracks = new_tracks
        return list(self.tracks.items())


class GstByteTrack(GstBase.BaseTransform):
    """
    GStreamer element that reads object detections and applies ByteTrack.

    This element parses GstAnalyticsRelationMeta attached by the upstream
    yoloxtensordec element, applies a simple tracking algorithm,
    and updates the tracking metadata in-place.
    """

    __gtype_name__ = "GstByteTrack"

    # Class-level configurations, updated dynamically before pipeline creation
    tracker_type: str = "iou"
    verbose: bool = False

    __gstmetadata__ = (
        "ByteTrack Object Tracker",
        "Filter/Effect/Video/Tracker",
        "Tracks objects using ByteTrack on GstAnalyticsRelationMeta",
        "Author <jarno@ralli.fi>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw"),
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw"),
        ),
    )

    def __init__(self) -> None:
        """Initialize GstByteTrack. Tracker is lazily initialized on first frame."""
        super().__init__()
        self.tracker: Optional[Any] = None

    def do_transform_ip(self, buf: Gst.Buffer) -> Gst.FlowReturn:
        """
        Processes GstBuffers in-place to update analytics metadata with tracking IDs.

        Parameters
        ----------
        buf : Gst.Buffer
            The GstBuffer being passed through the element.

        Returns
        -------
        Gst.FlowReturn
            The Gst flow return value, indicating success or failure.
        """
        # 1. Lazy initialization of the tracker to extract exact negotiated framerate
        if self.tracker is None:
            fps: float = 30.0  # default fallback
            caps = self.sinkpad.get_current_caps()
            if caps:
                struct = caps.get_structure(0)
                success, num, denom = struct.get_fraction("framerate")
                if success and denom != 0:
                    fps = num / denom

            if self.verbose:
                print(f"Auto-negotiated video framerate: {fps:.2f} FPS")

            if self.tracker_type == "bytetrack":
                try:
                    global sv, np
                    import numpy as np
                    import supervision as sv
                except ImportError as e:
                    raise ImportError(
                        "Error: 'supervision' or 'numpy' package is not installed. "
                        "Please install 'supervision' inside your container or "
                        "use the default 'iou' tracker."
                    ) from e

                # Initialize Roboflow Supervision's production-ready ByteTrack with actual FPS
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=0.25,
                    lost_track_buffer=30,
                    minimum_matching_threshold=0.8,
                    frame_rate=int(fps),
                )
            else:
                self.tracker = SimpleTracker()

        # Retrieve GstAnalyticsRelationMeta from the buffer
        relation_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if not relation_meta:
            return Gst.FlowReturn.OK

        detections: List[List[float]] = []
        od_mtds: List[Any] = []

        # Extract all Object Detection descriptors sequentially, scanning up to the exact total count
        total_descriptors = GstAnalytics.relation_get_length(relation_meta)

        for idx in range(total_descriptors):
            success, od_mtd = relation_meta.get_od_mtd(idx)
            if not success:
                continue

            ok, x, y, w, h, conf = od_mtd.get_location()
            if ok:
                detections.append([float(x), float(y), float(w), float(h)])
                od_mtds.append(od_mtd)

        if not detections:
            return Gst.FlowReturn.OK

        # Process detections with selected tracker
        tracks: List[Tuple[int, List[float]]] = []

        if self.tracker_type == "bytetrack":
            xyxy_list: List[List[float]] = []
            conf_list: List[float] = []
            class_ids: List[int] = []

            for od_mtd in od_mtds:
                ok, x, y, w, h, conf = od_mtd.get_location()
                if ok:
                    # Convert to Pascal VOC format: [x_min, y_min, x_max, y_max]
                    xyxy_list.append([float(x), float(y), float(x + w), float(y + h)])
                    conf_list.append(float(conf))
                    class_ids.append(int(od_mtd.get_obj_type()))

            if not xyxy_list:
                return Gst.FlowReturn.OK

            sv_detections = sv.Detections(
                xyxy=np.array(xyxy_list, dtype=np.float32),
                confidence=np.array(conf_list, dtype=np.float32),
                class_id=np.array(class_ids, dtype=np.int32),
            )

            tracked_detections = self.tracker.update_with_detections(sv_detections)

            # Format tracks as a list of (track_id, [x, y, w, h])
            if tracked_detections.tracker_id is not None:
                for i in range(len(tracked_detections)):
                    xyxy = tracked_detections.xyxy[i]
                    track_id = int(tracked_detections.tracker_id[i])
                    track_box = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                    tracks.append((track_id, track_box))
        else:
            # Simple IoU tracker
            tracks = self.tracker.update(detections)

        # Map tracked IDs back and link to the Object Detection descriptor
        if self.verbose:
            print(
                f"\n--- Frame [PTS: {buf.pts}] {self.tracker_type.upper()} Tracking Update ---"
            )

        for track_id, track_box in tracks:
            best_match_idx: int = -1
            best_match_iou: float = 0.5

            for idx, od_mtd in enumerate(od_mtds):
                ok, x, y, w, h, conf = od_mtd.get_location()
                roi_box = [float(x), float(y), float(w), float(h)]

                # Simple IoU matching to identify which detection this track belongs to
                iou = compute_iou(track_box, roi_box)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = idx

            if best_match_idx != -1:
                matched_od = od_mtds[best_match_idx]
                label = GLib.quark_to_string(matched_od.get_obj_type())

                # Print the bounding boxes only if verbose option is enabled
                if self.verbose:
                    print(
                        f"Track ID {track_id} ({label}): x={track_box[0]:.1f}, y={track_box[1]:.1f}, w={track_box[2]:.1f}, h={track_box[3]:.1f}"
                    )

                # Add tracking descriptor
                ok, tracking_mtd = relation_meta.add_tracking_mtd(track_id, buf.pts)
                if ok:
                    # Relate tracking descriptor to detection descriptor
                    relation_meta.set_relation(
                        GstAnalytics.RelTypes.RELATE_TO, matched_od.id, tracking_mtd.id
                    )

        return Gst.FlowReturn.OK


def run_pipeline(
    video_file_path: str,
    backend: str = "nd-array",
    tracker: str = "iou",
    verbose: bool = False,
    box_threshold: float = 0.4,
    class_threshold: float = 0.4,
    iou_threshold: float = 0.7,
    model_type: str = "medium",
) -> None:
    """
    Configure, build, and execute the GStreamer tracking pipeline.

    Parameters
    ----------
    video_file_path : str
        The absolute or relative path to the input video file.
    backend : str, optional
        The Burn inference backend ('nd-array', 'vulkan', or 'cuda'), by default 'nd-array'.
    tracker : str, optional
        The tracking algorithm choice ('iou' or 'bytetrack'), by default 'iou'.
    verbose : bool, optional
        Whether to print verbose console outputs for active tracked objects, by default False.
    box_threshold : float, optional
        Box confidence threshold for yoloxtensordec, by default 0.4.
    class_threshold : float, optional
        Class confidence threshold for yoloxtensordec, by default 0.4.
    iou_threshold : float, optional
        NMS IoU threshold for yoloxtensordec, by default 0.7.
    model_type : str, optional
        YOLOX model type ('nano', 'tiny', 'small', 'medium', 'large', 'extra-large'), by default 'medium'.

    Raises
    ------
    RuntimeError
        If the input file does not exist.
    """
    if not os.path.exists(video_file_path):
        raise RuntimeError(f"Error: Input file '{video_file_path}' does not exist.")

    # Initialize GStreamer
    Gst.init(None)

    # Set tracker selection and verbosity on the GStreamer element class
    GstByteTrack.tracker_type = tracker
    GstByteTrack.verbose = verbose

    # Register the custom in-memory element
    Gst.Element.register(None, "gstbytetrack", Gst.Rank.NONE, GstByteTrack.__gtype__)

    # Build the pipeline with our custom element inserted after yoloxtensordec
    pipeline_definition = f"""
        filesrc location={video_file_path} !
        qtdemux ! h264parse ! avdec_h264 !
        videoconvertscale ! video/x-raw,width=800,height=640 !
        queue max-size-buffers=2 !
        burn-yoloxinference backend-type={backend} model-type={model_type} !
        queue max-size-buffers=2 !
        yoloxtensordec label-file=COCO_classes.txt
                     box-confidence-threshold={box_threshold}
                     class-confidence-threshold={class_threshold}
                     iou-threshold={iou_threshold} !
        gstbytetrack !
        videoconvertscale ! objectdetectionoverlay !
        videoconvertscale ! autovideosink sync=false
    """

    print("=== Pipeline Definition ===")
    print(pipeline_definition.strip())
    print("===========================")

    pipeline = Gst.parse_launch(pipeline_definition)

    # Create GLib mainloop and run
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle GStreamer bus messages."""
        if message.type == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"Error from {message.src.get_name()}: {err.message}")
            if dbg:
                print(f"Debug info: {dbg}")
            loop.quit()

    bus.connect("message", on_message)

    print("Starting pipeline... Press Ctrl+C to stop.")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GStreamer Python tracking pipeline with ByteTrack."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input video file."
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="nd-array",
        choices=["nd-array", "vulkan", "cuda"],
        help="Burn inference backend (default: nd-array).",
    )
    parser.add_argument(
        "-t",
        "--tracker",
        type=str,
        default="iou",
        choices=["iou", "bytetrack"],
        help="Tracker algorithm selection: 'iou' or 'bytetrack' (default: iou).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output printing active track IDs and positions.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.4,
        help="Box confidence threshold for yoloxtensordec (default: 0.4).",
    )
    parser.add_argument(
        "--class-threshold",
        type=float,
        default=0.4,
        help="Class confidence threshold for yoloxtensordec (default: 0.4).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="NMS IoU threshold for yoloxtensordec (default: 0.7).",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="small",
        choices=["nano", "tiny", "small", "medium", "large", "extra-large"],
        help="YOLOX model type (default: small).",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            args.input,
            args.backend,
            args.tracker,
            args.verbose,
            args.box_threshold,
            args.class_threshold,
            args.iou_threshold,
            args.model_type,
        )
    except Exception as e:
        print(e)
        sys.exit(1)
