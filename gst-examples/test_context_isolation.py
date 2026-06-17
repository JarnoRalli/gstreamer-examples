import gi
import numpy as np
import torch

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


Gst.init(None)

# We use CPU decoding in GStreamer (avdec_h264) to avoid any CUDA context conflicts
pipeline_str = (
    "filesrc location=/workspace/gst-examples/terminator2.mp4 ! "
    "qtdemux ! h264parse ! avdec_h264 ! "
    "videoconvertscale ! video/x-raw,width=800,height=640,format=RGBA ! "
    "fakesink name=sink"
)

pipeline = Gst.parse_launch(pipeline_str)
sink = pipeline.get_by_name("sink")

print("Initializing PyTorch on CUDA GPU...")
device = torch.device("cuda")
model = (
    torch.hub.load(
        "Megvii-BaseDetection/YOLOX", "yolox_s", pretrained=True, trust_repo=True
    )
    .to(device)
    .eval()
)
frame_count = 0


def on_buffer_probe(pad, info):
    global frame_count
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    caps = pad.get_current_caps()
    struct = caps.get_structure(0)
    width = struct.get_value("width")
    height = struct.get_value("height")

    success, map_info = buf.map(Gst.MapFlags.READ)
    if success:
        try:
            # Wrap mapped CPU buffer as numpy array
            numpy_array = np.ndarray(
                (height, width, 4), dtype=np.uint8, buffer=map_info.data
            )

            # Convert to PyTorch GPU tensor (fast host-to-device copy)
            torch_tensor = torch.from_numpy(numpy_array).to(device)

            # Preprocess
            rgb_tensor = torch_tensor[:, :, :3].permute(2, 0, 1).float() / 255.0
            batch_input = rgb_tensor.unsqueeze(0)

            # Run model on Blackwell GPU!
            with torch.no_grad():
                out = model(batch_input)

            print(
                f"Frame {frame_count}: Successfully executed YOLOX on GPU! Output shape: {out.shape}"
            )
            frame_count += 1
        finally:
            buf.unmap(map_info)

    if frame_count >= 10:
        GLib.idle_add(loop.quit)
    return Gst.PadProbeReturn.OK


sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_buffer_probe)

loop = GLib.MainLoop()
pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
