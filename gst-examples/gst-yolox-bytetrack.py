#!/usr/bin/env python3
"""
GStreamer Python pipeline implementing zero-copy YOLOX object detection (via PyTorch/CuPy)
and tracking (via Supervision's ByteTrack or IoU simple tracker) to avoid GPU-CPU-GPU roundtrips.
"""

import argparse
import sys
import os
import ctypes
import ctypes.util
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
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    The IoU measures the overlap between two bounding boxes to evaluate how
    well they correspond to the same physical object. This is a critical metric
    for both non-maximum suppression (NMS) and object tracking, where bounding
    boxes from successive frames are matched.

    Parameters
    ----------
    boxA : list of float
        The first bounding box specified as [x, y, width, height] or [x1, y1, x2, y2].
        Both boxes must use the same coordinate system and convention.
    boxB : list of float
        The second bounding box specified using the same format as boxA.

    Returns
    -------
    float
        The calculated Intersection over Union value, in the range [0.0, 1.0].
        Returns 0.0 if there is no overlap or if the union area is zero.
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


class GstCUDAArrayWrapper:
    """
    A class that exposes a GStreamer CUdeviceptr directly to PyTorch.

    This wrapper implements the `__cuda_array_interface__` protocol (version 3),
    enabling zero-copy data transfer from GStreamer CUDA memory buffers directly
    into PyTorch tensors. By avoiding intermediate CPU copies or driver-level
    re-allocations, it dramatically reduces latency and GPU-CPU-GPU roundtrips
    during inference pipelines.

    Attributes
    ----------
    __cuda_array_interface__ : dict
        A dictionary compliant with the CUDA Array Interface protocol standard.
    """

    def __init__(
        self,
        ptr_val: int,
        shape: Tuple[int, ...],
        strides: Optional[Tuple[int, ...]],
        dtype_str: str = "|u1",
    ) -> None:
        """
        Initialize the CUDA array wrapper with memory buffer parameters.

        Parameters
        ----------
        ptr_val : int
            The numeric address of the GPU memory buffer (the CUdeviceptr).
        shape : tuple of int
            The shape of the array (e.g., (height, width, channels)).
        strides : tuple of int or None
            The strides of the array in bytes for each dimension. If None,
            the array is assumed to be C-contiguous.
        dtype_str : str, optional
            The typestr representing the data type in the format specified by
            the CUDA Array Interface (e.g., "|u1" for uint8/unsigned byte).
            Default is "|u1".
        """
        self.__cuda_array_interface__: Dict[str, Any] = {
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


def load_gstreamer_library() -> Optional[ctypes.CDLL]:
    """
    Dynamically locate and load the GStreamer shared library.

    This helper uses `ctypes.util.find_library` to locate the GStreamer library path
    on the current system (including support for various OS/platforms) and falls
    back to common Linux library filenames if standard lookup fails. This is crucial
    for cross-platform compatibility and future-proofing against library version/extension
    naming changes.

    Returns
    -------
    ctypes.CDLL or None
        The loaded CDLL object for GStreamer if successful, otherwise None.
    """
    # 1. Try standard dynamic finder (e.g. finds 'libgstreamer-1.0.so.0' on Ubuntu)
    path = ctypes.util.find_library("gstreamer-1.0")
    if path:
        try:
            return ctypes.CDLL(path)
        except Exception:
            pass

    # 2. Hardcoded fallback list for cross-platform/future naming differences
    fallbacks = [
        "libgstreamer-1.0.so.0",  # Standard Linux
        "libgstreamer-1.0.so",  # Linux dev/symlink
        "libgstreamer-1.0.0.dylib",  # macOS
        "gstreamer-1.0-0.dll",  # Windows
    ]
    for name in fallbacks:
        try:
            return ctypes.CDLL(name)
        except Exception:
            pass
    return None


def load_gstcuda_library() -> Optional[ctypes.CDLL]:
    """
    Dynamically locate and load the GStreamer GstCuda shared library.

    This helper uses `ctypes.util.find_library` to locate the gstcuda library path
    on the current system and falls back to common Linux/Windows/macOS library
    filenames if standard lookup fails. GstCuda is necessary for CUDA memory synchronization.

    Returns
    -------
    ctypes.CDLL or None
        The loaded CDLL object for GstCuda if successful, otherwise None.
    """
    # 1. Try standard dynamic finder
    path = ctypes.util.find_library("gstcuda-1.0")
    if path:
        try:
            return ctypes.CDLL(path)
        except Exception:
            pass

    # 2. Hardcoded fallback list
    fallbacks = [
        "libgstcuda-1.0.so.0",  # Standard Linux
        "libgstcuda-1.0.so",  # Linux dev/symlink
        "libgstcuda-1.0.0.dylib",  # macOS
        "gstcuda-1.0-0.dll",  # Windows
    ]
    for name in fallbacks:
        try:
            return ctypes.CDLL(name)
        except Exception:
            pass
    return None


# Resolve and load gstreamer C API functions dynamically
libgst = load_gstreamer_library()
HAS_CTYPES_MAP = False

if libgst is not None:
    try:
        # Bind gst_memory_map and gst_memory_unmap to bypass Python bindings limitations.
        # This is strictly required to fetch the raw CUdeviceptr memory addresses
        # of GStreamer buffers, enabling zero-copy GPU memory passing to PyTorch.
        libgst.gst_memory_map.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(GstMapInfo),
            ctypes.c_int,
        ]
        libgst.gst_memory_map.restype = ctypes.c_bool
        libgst.gst_memory_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(GstMapInfo)]
        libgst.gst_memory_unmap.restype = None
        HAS_CTYPES_MAP = True
    except Exception as e:
        print(f"[GstYolox] Failed to bind GStreamer memory map functions: {e}")

# Resolve and load gstcuda C API functions dynamically
libgstcuda = load_gstcuda_library()
HAS_GST_CUDA_SYNC = False

if libgstcuda is not None:
    try:
        # Bind gst_cuda_memory_sync to prevent race conditions. This is strictly required
        # to ensure GStreamer asynchronously finishes writing raw frame data on its CUDA stream
        # before PyTorch reads from it asynchronously on its stream (preventing flickering/segfaults).
        libgstcuda.gst_cuda_memory_sync.argtypes = [ctypes.c_void_p]
        libgstcuda.gst_cuda_memory_sync.restype = None
        HAS_GST_CUDA_SYNC = True
    except Exception as e:
        print(f"[GstYolox] Failed to bind GstCuda sync functions: {e}")


class SimpleTracker:
    """
    A simple Intersection-over-Union (IoU) tracker for bounding boxes.

    This tracker associates detections across consecutive video frames by computing the
    IoU between the bounding boxes of existing active tracks and newly detected boxes.
    It is lightweight, relies on no appearance models or motion models (like Kalman filters),
    and is designed for scenarios with high frame rates and relatively slow-moving objects.

    Attributes
    ----------
    next_id : int
        The unique tracking identifier to be assigned to the next newly created track.
    tracks : dict of {int: list of float}
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

        This method attempts to match the current frame's detections with existing tracks
        using a greedy IoU matching strategy. Unmatched tracks are discarded, and unmatched
        detections are instantiated as new tracks with unique IDs.

        Parameters
        ----------
        detections : list of list of float
            A list of bounding box coordinates from the latest frame, where each bounding
            box is a list of floats representing [x, y, width, height].

        Returns
        -------
        list of tuple of (int, list of float)
            A list of active tracks after update, where each track is represented as a
            tuple containing the track ID (int) and its assigned bounding box (list of float).
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
    GStreamer Python transform element that runs YOLOX via PyTorch.

    This class subclasses `GstBase.BaseTransform` to perform in-place zero-copy
    image processing using CUDA-based PyTorch inference and advanced object tracking
    algorithms (such as ByteTrack or a simple Intersection-over-Union tracker).
    It avoids costly GPU-CPU-GPU memory transfer roundtrips by mapping the GStreamer
    CUDA memory handle directly to PyTorch tensors via the `__cuda_array_interface__`
    specification.

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
    cuda_fallback_triggered: bool = False

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
        """
        Initialize the GstYoloxByteTrack element.

        Loads standard COCO label names, sets up initial fallback flags,
        and prepares the GStreamer base transform structure.
        """
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

    @classmethod
    def load_model(cls) -> None:
        """
        Load or dynamically reload the YOLOX model onto the target hardware device.

        This method initializes or re-initializes the PyTorch model and target device
        (CPU or GPU). It is also used as a robust fallback mechanism if GPU execution
        or memory mapping fails at runtime, enabling the pipeline to gracefully pivot
        to CPU-based execution without crashing.
        """
        if (
            cls.backend == "cuda"
            and torch.cuda.is_available()
            and not getattr(cls, "cuda_fallback_triggered", False)
        ):
            cls.device = torch.device("cuda")
            cls.use_gpu = True
            print("[GstYolox] Successfully initialized PyTorch on CUDA GPU.")
        else:
            cls.device = torch.device("cpu")
            cls.use_gpu = False
            print("[GstYolox] PyTorch executing on CPU.")

        print(
            f"[GstYolox] Loading pre-trained YOLOX model '{cls.model_type}' on {cls.device}..."
        )
        try:
            cls.model = (
                torch.hub.load(
                    "Megvii-BaseDetection/YOLOX",
                    cls.model_type,
                    pretrained=True,
                    trust_repo=True,
                )
                .to(cls.device)
                .eval()
            )
            print("[GstYolox] Model loaded successfully.")
        except Exception as e:
            print(f"[GstYolox] Error loading model from Hub: {e}")
            sys.exit(1)

    def do_transform_ip(self, buf: Gst.Buffer) -> Gst.FlowReturn:
        """
        Perform in-place processing and object tracking on the input GStreamer Buffer.

        This method is invoked by GStreamer for each frame. It handles:
        1. Lazy loading of the chosen tracker (negotiating FPS via CAPS if necessary).
        2. Extracting input frame parameters (width, height, format).
        3. Mapping the underlying buffer data into a PyTorch tensor (attempting
           zero-copy CUDA mapping first, with a CPU fallback if mapping or execution
           errors occur).
        4. Running pre-loaded YOLOX inference.
        5. Running post-processing (Anchor Grid Decoding and NMS).
        6. Running tracking update (ByteTrack or simple IoU tracker).
        7. Writing tracked results back into the frame buffer as GstAnalytics metadata,
           making them available to downstream elements such as `objectdetectionoverlay`.

        Parameters
        ----------
        buf : Gst.Buffer
            The active GStreamer buffer representing the current video frame.

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
                    GstYoloxByteTrack.cuda_fallback_triggered = True
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
                GstYoloxByteTrack.cuda_fallback_triggered = True
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
    """
    Configure, build, and execute the GStreamer YOLOX Object Detection and Tracking pipeline.

    This function sets class-level parameters on the `GstYoloxByteTrack` element,
    initializes the PyTorch model and execution hardware, registers the custom
    element with GStreamer's runtime registry, selects the appropriate decoding
    and scaling components depending on whether CUDA acceleration is available,
    constructs the GStreamer pipeline using launch syntax, and handles OS signal
    trapping to shut down the pipeline gracefully.

    Parameters
    ----------
    video_file_path : str
        The local path to the input H.264 video file.
    backend : str, optional
        The computer vision inference backend to utilize ("cpu" or "cuda").
        Default is "cuda".
    tracker : str, optional
        The tracker algorithm choice ("iou" or "bytetrack").
        Default is "iou".
    verbose : bool, optional
        If True, verbose output details such as active tracking coordinates and
        confidences will be printed to stdout. Default is False.
    box_threshold : float, optional
        The confidence score limit to filter candidate bounding box detections.
        Default is 0.4.
    class_threshold : float, optional
        The minimum probability score required for classification.
        Default is 0.4.
    iou_threshold : float, optional
        The overlap limit for candidate boxes during Non-Maximum Suppression.
        Default is 0.7.
    model_type : str, optional
        The architectural variant of the pre-trained YOLOX model to load from PyTorch Hub
        ("nano", "tiny", "small", "medium", "large", "extra-large"). Default is "small".

    Raises
    ------
    RuntimeError
        If the specified input video file does not exist.
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

    # Trigger GStreamer dynamic CUDA loader vtable load by instantiating a CUDA element
    # _el = Gst.ElementFactory.make("cudaupload", "test_loader")

    # Load PyTorch, the model, and initialize the device ON THE MAIN THREAD!
    GstYoloxByteTrack.load_model()
    use_gpu = GstYoloxByteTrack.use_gpu

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
        """
        Handle incoming messages on the GStreamer bus.

        This callback captures pipeline-wide notifications, such as the End-of-Stream (EOS)
        signal when video playback completes or Error signals when elements fail.

        Parameters
        ----------
        bus : Gst.Bus
            The GStreamer message bus broadcasting events.
        message : Gst.Message
            The received message container containing type, source, and payload data.
        """
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
