#!/usr/bin/env python3
"""
GStreamer Python pipeline implementing zero-copy YOLOX object detection (via PyTorch/CuPy)
and tracking (via Supervision's ByteTrack or IoU simple tracker) to avoid GPU-CPU-GPU roundtrips.
"""

import argparse
import sys
import os
import ctypes
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

# Attempt loading GstCuda if available
try:
    gi.require_version("GstCuda", "1.0")
    from gi.repository import GstCuda

    HAS_GST_CUDA = True
except Exception:
    HAS_GST_CUDA = False


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
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


class GstCUDAArrayWrapper:
    """
    Exposes GStreamer CUdeviceptr directly to PyTorch
    via the __cuda_array_interface__ protocol (Zero-Copy).
    """

    def __init__(self, ptr_val, shape, strides, dtype_str="|u1"):
        self.__cuda_array_interface__ = {
            "version": 3,
            "data": (ptr_val, False),  # (pointer address, read_only)
            "shape": shape,
            "typestr": dtype_str,  # "|u1" represents uint8 (unsigned 1-byte)
            "strides": strides,
        }


class GstMapInfo(ctypes.Structure):
    _fields_ = [
        ("memory", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("padding", ctypes.c_int),  # Padding to align 64-bit pointers
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("maxsize", ctypes.c_size_t),
        ("user_data", ctypes.c_void_p * 4),
        ("_gst_reserved", ctypes.c_void_p * 4),  # GStreamer ABI padding
    ]


try:
    libgst = ctypes.CDLL("libgstreamer-1.0.so.0")
    libgst.gst_memory_map.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(GstMapInfo),
        ctypes.c_int,
    ]
    libgst.gst_memory_map.restype = ctypes.c_bool
    libgst.gst_memory_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(GstMapInfo)]
    libgst.gst_memory_unmap.restype = None
    HAS_CTYPES_MAP = True
except Exception:
    HAS_CTYPES_MAP = False


try:
    libgstcuda = ctypes.CDLL("libgstcuda-1.0.so.0")
    libgstcuda.gst_cuda_memory_sync.argtypes = [ctypes.c_void_p]
    libgstcuda.gst_cuda_memory_sync.restype = None
    HAS_GST_CUDA_SYNC = True
except Exception:
    HAS_GST_CUDA_SYNC = False


class SimpleTracker:
    """A simple Intersection-over-Union (IoU) tracker for bounding boxes."""

    def __init__(self) -> None:
        self.next_id: int = 1
        self.tracks: Dict[int, List[float]] = {}

    def update(self, detections: List[List[float]]) -> List[Tuple[int, List[float]]]:
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
    GStreamer Python transform element that runs YOLOX via PyTorch
    (with zero-copy CUDA memory) and applies object tracking.
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
        "Zero-Copy YOLOX + ByteTrack Element",
        "Filter/Effect/Video/Inference/Tracker",
        "Performs high-performance zero-copy PyTorch inference and ByteTrack object tracking",
        "Author <jarno@ralli.fi>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "video/x-raw(memory:CUDAMemory), format=RGBA; video/x-raw, format=RGBA; video/x-raw, format=RGB"
            ),
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "video/x-raw(memory:CUDAMemory), format=RGBA; video/x-raw, format=RGBA; video/x-raw, format=RGB"
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.labels: List[str] = []
        self.cuda_fallback_triggered: bool = False

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
        mem = buf.peek_memory(0)
        is_cuda = HAS_GST_CUDA and GstCuda.is_cuda_memory(mem)

        # 3. Extract RGB tensor via CUDA Zero-Copy (if active and not falling back)
        if (
            is_cuda
            and self.use_gpu
            and not self.cuda_fallback_triggered
            and HAS_CTYPES_MAP
        ):
            if HAS_GST_CUDA_SYNC:
                libgstcuda.gst_cuda_memory_sync(hash(mem))
            map_info = GstMapInfo()
            # 131073 = GST_MAP_READ (1) | GST_MAP_CUDA (131072)
            success = libgst.gst_memory_map(hash(mem), ctypes.byref(map_info), 131073)
            if success:
                try:
                    ptr_val = map_info.data
                    if ptr_val:
                        # Calculate the real hardware-pitched stride
                        actual_stride = buf.get_size() // height
                        cuda_wrapper = GstCUDAArrayWrapper(
                            ptr_val=ptr_val,
                            shape=(height, width, 4),
                            strides=(actual_stride, 4, 1),
                            dtype_str="|u1",
                        )
                        torch_tensor = torch.as_tensor(cuda_wrapper, device=self.device)
                        rgb_tensor = torch_tensor[:, :, :3].permute(2, 0, 1).float()
                except Exception as e:
                    print(
                        f"[GstYolox] CUDA zero-copy mapping failed: {e}. Switching to CPU fallback."
                    )
                    self.cuda_fallback_triggered = True
                finally:
                    libgst.gst_memory_unmap(hash(mem), ctypes.byref(map_info))

        # 4. Fallback: map to CPU system memory
        if rgb_tensor is None:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                try:
                    # Wrap the mapped system buffer into NumPy array (Zero-Copy CPU)
                    # For RGBA caps, pixels are 4-byte. For RGB, pixels are 3-byte.
                    channels = 4 if "RGBA" in struct.get_string("format") else 3
                    numpy_array = np.ndarray(
                        (height, width, channels), dtype=np.uint8, buffer=map_info.data
                    ).copy()
                    torch_tensor = torch.from_numpy(numpy_array).to(self.device)
                    # Preprocess RGB
                    rgb_tensor = torch_tensor[:, :, :3].permute(2, 0, 1).float()
                finally:
                    buf.unmap(map_info)

        if rgb_tensor is None:
            print(
                "[GstYolox] Error: Failed to acquire any frame data. Passing buffer through."
            )
            return Gst.FlowReturn.OK

        # 5. YOLOX Model Inference
        batch_input = rgb_tensor.unsqueeze(0)  # [1, 3, height, width]

        try:
            with torch.no_grad():
                predictions = self.model(batch_input)
        except Exception as e:
            # If a Blackwell GPU execution error happens, fallback immediately to CPU
            if "CUDA error" in str(e) or "kernel image" in str(e):
                print(f"[GstYolox] PyTorch GPU kernel execution failed: {e}")
                print(
                    "[GstYolox] Disabling GPU path permanently for this session and falling back to CPU."
                )
                self.cuda_fallback_triggered = True
                self.load_model()
                # Run again on CPU
                batch_input = batch_input.to(self.device)
                with torch.no_grad():
                    predictions = self.model(batch_input)
            else:
                print(f"[GstYolox] Inference error: {e}")
                return Gst.FlowReturn.OK

        # 6. Post-process (Anchor Grid Decode + NMS)
        from yolox.utils import postprocess

        detections = postprocess(
            predictions, 80, self.box_threshold, self.iou_threshold
        )

        # 7. Tracking (ByteTrack or IoU)
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

        # 8. Attach Metadata using GstAnalytics API so 'objectdetectionoverlay' draws it
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
    """Configures and runs the PyTorch YOLOX GStreamer pipeline."""
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

    # Trigger GStreamer dynamic CUDA loader vtable load by instantiating a CUDA element
    # _el = Gst.ElementFactory.make("cudaupload", "test_loader")

    # Load PyTorch, the model, and initialize the device ON THE MAIN THREAD!
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

    # Dynamic pipeline decoding and scaling selection based on backend choice
    if backend == "cuda" and use_gpu:
        print(
            f"[Pipeline] Initializing GPU pipeline (nvh264dec + cudaconvertscale) with PyTorch YOLOX inference on {backend.upper()}..."
        )
        decode_and_scale = """
            nvh264dec !
            cudaconvertscale ! video/x-raw(memory:CUDAMemory),width=800,height=640,format=RGBA !
        """
        download_and_overlay = """
            cudadownload ! video/x-raw,format=RGBA !
            objectdetectionoverlay !
            videoconvertscale ! autovideosink sync=false
        """
    else:
        print(
            f"[Pipeline] Initializing CPU pipeline (avdec_h264 + videoconvertscale) with PyTorch YOLOX inference on {backend.upper()}..."
        )
        decode_and_scale = """
            avdec_h264 !
            videoconvertscale ! video/x-raw,width=800,height=640,format=RGBA !
        """
        download_and_overlay = """
            objectdetectionoverlay !
            videoconvertscale ! autovideosink sync=false
        """

    pipeline_definition = f"""
        filesrc location={video_file_path} !
        qtdemux ! h264parse !
        {decode_and_scale}
        queue max-size-buffers=2 !
        gstyoloxbytetrack !
        queue max-size-buffers=2 !
        {download_and_overlay}
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
        description="GStreamer Python tracking pipeline with zero-copy PyTorch YOLOX and ByteTrack."
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
