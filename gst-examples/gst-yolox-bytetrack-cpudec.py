#!/usr/bin/env python3
"""
GStreamer Python pipeline implementing CPU decoding and GPU/CPU YOLOX object detection (via PyTorch)
and tracking (via Supervision's ByteTrack or IoU simple tracker).
"""

import argparse
import sys
import os
from typing import List, Tuple, Dict, Any, Optional

import torch
import numpy as np
import supervision as sv

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstAnalytics", "1.0")
from gi.repository import Gst, GstBase, GstAnalytics, GLib  # noqa: E402

# Initialize GStreamer
Gst.init(None)


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    boxA : List[float]
        The first bounding box specified as [x, y, width, height].
    boxB : List[float]
        The second bounding box specified as [x, y, width, height].

    Returns
    -------
    float
        The calculated Intersection over Union value, in the range [0.0, 1.0].
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

    Attributes
    ----------
    next_id : int
        The unique tracking identifier to be assigned to the next newly created track.
    tracks : Dict[int, List[float]]
        A mapping of active track IDs to their latest bounding box coordinates
        represented as [x, y, width, height].
    """

    def __init__(self) -> None:
        """Initialize the SimpleTracker with an empty list of active tracks."""
        self.next_id: int = 1
        self.tracks: Dict[int, List[float]] = {}

    def update(self, detections: List[List[float]]) -> List[Tuple[int, List[float]]]:
        """
        Update the tracker state with a new set of bounding box detections.

        Parameters
        ----------
        detections : List[List[float]]
            A list of bounding box coordinates from the latest frame, where each bounding
            box is a list of floats representing [x, y, width, height].

        Returns
        -------
        List[Tuple[int, List[float]]]
            A list of active tracks after update, where each track is represented as a
            tuple containing the track ID (int) and its assigned bounding box (List[float]).
        """
        new_tracks: Dict[int, List[float]] = {}
        matched_detections = set()

        for track_id, last_box in self.tracks.items():
            best_iou = 0.3
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

        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                new_tracks[self.next_id] = det
                self.next_id += 1

        self.tracks = new_tracks
        return list(self.tracks.items())


class GstYoloxByteTrack(GstBase.BaseTransform):
    """
    GStreamer Python transform element that runs YOLOX via PyTorch on CPU-decoded frames.

    Attributes
    ----------
    model : torch.nn.Module or None
        The pre-loaded YOLOX PyTorch model instance used for inference.
    tracker : object or None
        The initialized tracker object (either a `supervision.ByteTrack` or
        a `SimpleTracker` instance).
    device : torch.device or None
        The PyTorch hardware device context (CUDA or CPU) on which tensor calculations
        and model inference are performed.
    use_gpu : bool
        Indicates whether PyTorch inference is set up to run on the GPU.
    model_type : str
        The specific YOLOX variant to load (e.g. 'yolox_nano', 'yolox_tiny',
        'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x').
    backend : str
        The preferred inference backend, either "cuda" or "cpu".
    tracker_type : str
        The object tracking algorithm choice, either "bytetrack" or "iou".
    verbose : bool
        If True, enables logging of inference speed, tracking IDs, coordinates,
        and other debug properties on stdout.
    box_threshold : float
        The minimum confidence score required to keep a detected bounding box.
    class_threshold : float
        The minimum class probability score required for classification.
    iou_threshold : float
        The Intersection over Union threshold used in PyTorch's Non-Maximum
        Suppression (NMS) stage.
    """

    __gtype_name__ = "GstYoloxByteTrack"

    # Pre-loaded model and devices on main thread
    model: Optional[Any] = None
    tracker: Optional[Any] = None
    device: Optional[Any] = None
    use_gpu: bool = False

    # Static properties set dynamically
    model_type: str = "yolox_s"
    backend: str = "cuda"
    tracker_type: str = "iou"
    verbose: bool = False
    box_threshold: float = 0.4
    class_threshold: float = 0.4
    iou_threshold: float = 0.7

    __gstmetadata__ = (
        "CPU Decode YOLOX + ByteTrack Element",
        "Filter/Effect/Video/Inference/Tracker",
        "Performs high-performance PyTorch inference on CPU-decoded frames and ByteTrack object tracking",
        "Author <jarno@ralli.fi>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw, format=RGBA; video/x-raw, format=RGB"),
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw, format=RGBA; video/x-raw, format=RGB"),
        ),
    )

    def __init__(self) -> None:
        """Initialize the element, loading COCO labels."""
        super().__init__()
        self.labels: List[str] = []

        # Load standard labels
        script_dir = os.path.dirname(os.path.abspath(__file__))
        label_file = os.path.join(script_dir, "COCO_classes.txt")
        if not os.path.exists(label_file):
            label_file = "/workspace/COCO_classes.txt"

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                self.labels = [line.strip() for line in f if line.strip()]
        else:
            self.labels = [f"class_{i}" for i in range(80)]

    def do_transform_ip(self, buf: Gst.Buffer) -> Gst.FlowReturn:
        """
        Perform in-place processing and object tracking on the input CPU Buffer.

        Parameters
        ----------
        buf : Gst.Buffer
            The active GStreamer buffer representing the current video frame in system memory.

        Returns
        -------
        Gst.FlowReturn
            An enum status indicating success (`Gst.FlowReturn.OK`) or failure.
        """
        # 1. Lazily load tracker
        if self.tracker is None:
            # Negotiate framerate for Supervision tracker
            fps = 30.0
            caps = self.sinkpad.get_current_caps()
            if caps:
                struct = caps.get_structure(0)
                success, num, denom = struct.get_fraction("framerate")
                if success and denom != 0:
                    fps = num / denom

            if self.tracker_type == "bytetrack":
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.box_threshold,
                    lost_track_buffer=30,
                    minimum_matching_threshold=0.8,
                    frame_rate=int(fps),
                )
            else:
                self.tracker = SimpleTracker()

        # 2. Get width, height and format of current frame
        caps = self.sinkpad.get_current_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        rgb_tensor = None

        # 3. Map buffer to CPU system memory and load to PyTorch device (Host-to-Device Copy if device is GPU)
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                # Wrap the mapped system buffer into NumPy array (Zero-Copy CPU-side)
                channels = 4 if "RGBA" in struct.get_string("format") else 3
                numpy_array = np.ndarray(
                    (height, width, channels), dtype=np.uint8, buffer=map_info.data
                )
                # Convert to PyTorch tensor and move to target device (GPU or CPU)
                torch_tensor = torch.from_numpy(numpy_array).to(self.device)
                # Preprocess RGB (slice Alpha if RGBA, permute to BCHW-compatible format, cast to float)
                rgb_tensor = torch_tensor[:, :, :3].permute(2, 0, 1).float()
            finally:
                buf.unmap(map_info)

        if rgb_tensor is None:
            print(
                "[GstYolox] Error: Failed to acquire any frame data. Passing buffer through."
            )
            return Gst.FlowReturn.OK

        # 4. YOLOX Model Inference
        batch_input = rgb_tensor.unsqueeze(0)  # [1, 3, height, width]

        try:
            with torch.no_grad():
                predictions = self.model(batch_input)
        except Exception as e:
            print(f"[GstYolox] Inference error: {e}")
            return Gst.FlowReturn.OK

        # 5. Post-process (Anchor Grid Decode + NMS)
        from yolox.utils import postprocess

        detections = postprocess(
            predictions, 80, self.box_threshold, self.iou_threshold
        )

        # 6. Tracking (ByteTrack or IoU)
        xyxy_list: List[List[float]] = []
        conf_list: List[float] = []
        class_ids: List[int] = []

        if detections[0] is not None:
            det_tensor = detections[0].cpu().numpy()
            xyxy_list = det_tensor[:, :4].tolist()
            conf_list = det_tensor[:, 4].tolist()
            class_ids = det_tensor[:, 6].astype(int).tolist()

        tracks: List[Tuple[int, List[float]]] = []

        if self.tracker_type == "bytetrack":
            if len(xyxy_list) > 0:
                sv_detections = sv.Detections(
                    xyxy=np.array(xyxy_list, dtype=np.float32),
                    confidence=np.array(conf_list, dtype=np.float32),
                    class_id=np.array(class_ids, dtype=np.int32),
                )
            else:
                sv_detections = sv.Detections.empty()

            tracked_detections = self.tracker.update_with_detections(sv_detections)
            if tracked_detections.tracker_id is not None:
                for i in range(len(tracked_detections)):
                    xyxy = tracked_detections.xyxy[i]
                    track_id = int(tracked_detections.tracker_id[i])
                    # Convert xyxy -> xywh
                    track_box = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                    tracks.append((track_id, track_box))
        else:
            # Simple IoU tracker requires xywh format
            xywh_list = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in xyxy_list
            ]
            tracks = self.tracker.update(xywh_list)

        # 7. Attach Metadata using GstAnalytics API so 'objectdetectionoverlay' draws it
        relation_meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if not relation_meta:
            relation_meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)

        if self.verbose:
            print(f"\n--- YOLOX Inference + {self.tracker_type.upper()} ---")

        # Map tracker outputs to detection objects
        for track_id, track_box in tracks:
            # Find closest detection match via IoU
            best_match_idx = -1
            best_match_iou = 0.5

            for idx, box in enumerate(xyxy_list):
                xywh_det = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                iou = compute_iou(track_box, xywh_det)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = idx

            if best_match_idx != -1:
                class_id = class_ids[best_match_idx]
                conf = conf_list[best_match_idx]
                label_name = (
                    self.labels[class_id] if class_id < len(self.labels) else "unknown"
                )
                label_quark = GLib.quark_from_string(label_name)

                # Add Object Detection metadata (xywh format)
                # Convert to integer coordinates for GstAnalytics format
                x_int, y_int, w_int, h_int = map(int, track_box)

                success, od_mtd = relation_meta.add_od_mtd(
                    label_quark, x_int, y_int, w_int, h_int, float(conf)
                )

                if success:
                    # Add Tracking metadata and relate it
                    ok, tracking_mtd = relation_meta.add_tracking_mtd(track_id, buf.pts)
                    if ok:
                        relation_meta.set_relation(
                            GstAnalytics.RelTypes.RELATE_TO, od_mtd.id, tracking_mtd.id
                        )

                        if self.verbose:
                            print(
                                f"Track ID {track_id} ({label_name}): x={x_int}, y={y_int}, w={w_int}, h={h_int} (conf: {conf:.2f})"
                            )

        return Gst.FlowReturn.OK


def run_pipeline(
    video_file_path: str,
    backend: str = "cuda",
    tracker: str = "iou",
    verbose: bool = False,
    box_threshold: float = 0.4,
    class_threshold: float = 0.4,
    iou_threshold: float = 0.7,
    model_type: str = "small",
) -> None:
    """
    Configures and runs the PyTorch YOLOX GStreamer pipeline with CPU decoding.

    Parameters
    ----------
    video_file_path : str
        The local path to the input H.264 video file.
    backend : str, optional
        The computer vision inference backend to utilize ("cpu" or "cuda").
    tracker : str, optional
        The tracker algorithm choice ("iou" or "bytetrack").
    verbose : bool, optional
        If True, verbose output details will be printed to stdout.
    box_threshold : float, optional
        The confidence score limit to filter candidate bounding box detections.
    class_threshold : float, optional
        The minimum probability score required for classification.
    iou_threshold : float, optional
        The overlap limit for candidate boxes during Non-Maximum Suppression.
    model_type : str, optional
        The architectural variant of the pre-trained YOLOX model to load from PyTorch Hub
        ("nano", "tiny", "small", "medium", "large", "extra-large").
    """
    if not os.path.exists(video_file_path):
        raise RuntimeError(f"Error: Input file '{video_file_path}' does not exist.")

    # Register custom Python YOLOX tracking element
    GstYoloxByteTrack.backend = backend
    GstYoloxByteTrack.tracker_type = tracker
    GstYoloxByteTrack.verbose = verbose
    GstYoloxByteTrack.box_threshold = box_threshold
    GstYoloxByteTrack.class_threshold = class_threshold
    GstYoloxByteTrack.iou_threshold = iou_threshold

    # Map model types
    model_mapping = {
        "nano": "yolox_nano",
        "tiny": "yolox_tiny",
        "small": "yolox_s",
        "medium": "yolox_m",
        "large": "yolox_l",
        "extra-large": "yolox_x",
    }
    model_type_str = model_mapping.get(model_type, "yolox_s")
    GstYoloxByteTrack.model_type = model_type_str

    # Load PyTorch, the YOLOX model, and initialize the target device
    if backend == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu = True
        print("[Pipeline] Successfully initialized PyTorch on CUDA GPU.")
    else:
        device = torch.device("cpu")
        use_gpu = False
        print("[Pipeline] PyTorch executing on CPU.")

    print(
        f"[Pipeline] Loading pre-trained YOLOX model '{model_type_str}' on {device}..."
    )
    try:
        model = (
            torch.hub.load(
                "Megvii-BaseDetection/YOLOX",
                model_type_str,
                pretrained=True,
                trust_repo=True,
            )
            .to(device)
            .eval()
        )
        print("[Pipeline] Model loaded successfully.")
    except Exception as e:
        print(f"[Pipeline] Error loading model from Hub: {e}")
        sys.exit(1)

    # Assign pre-loaded variables as class attributes
    GstYoloxByteTrack.model = model
    GstYoloxByteTrack.device = device
    GstYoloxByteTrack.use_gpu = use_gpu

    # Register element
    Gst.Element.register(
        None, "gstyoloxbytetrack", Gst.Rank.NONE, GstYoloxByteTrack.__gtype__
    )

    print(
        f"[Pipeline] Initializing pipeline with GStreamer decoding on CPU and PyTorch YOLOX inference on {backend.upper()}..."
    )
    pipeline_definition = f"""
        filesrc location={video_file_path} !
        qtdemux ! h264parse ! avdec_h264 !
        videoconvertscale ! video/x-raw,width=800,height=640,format=RGBA !
        queue max-size-buffers=2 !
        gstyoloxbytetrack !
        queue max-size-buffers=2 !
        objectdetectionoverlay !
        videoconvertscale ! autovideosink sync=false
    """

    print("=== Pipeline Definition ===")
    print(pipeline_definition.strip())
    print("===========================")

    pipeline = Gst.parse_launch(pipeline_definition)
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus: Gst.Bus, message: Gst.Message) -> None:
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
        description="GStreamer Python tracking pipeline with CPU-decoded PyTorch YOLOX and ByteTrack."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input video file."
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference backend (default: cuda).",
    )
    parser.add_argument(
        "-t",
        "--tracker",
        type=str,
        default="iou",
        choices=["iou", "bytetrack"],
        help="Tracker algorithm selection (default: iou).",
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
        help="Box confidence threshold (default: 0.4).",
    )
    parser.add_argument(
        "--class-threshold",
        type=float,
        default=0.4,
        help="Class confidence threshold (default: 0.4).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7).",
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
