"""
This example shows how to capture frames form a Gst-pipeline and process them with PyTorch.

For help regarding the command line arguments, run:
python gst-pytorch-example-1.py --help
"""

import gi
import numpy as np
import torch
import torchvision.transforms as T
import argparse
import contextlib
import time
from functools import partial

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

frame_format, pixel_bytes, model_precision = "RGBA", 4, "fp32"
ssd_utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)
start_time, frames_processed = None, 0


@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


def on_frame_probe(
    pad_in, info_in, detector_in, transform_in, device_in, detection_threshold_in
):
    """
    This function is called every time a new frame is available.
    :param pad_in: pad of the probe
    :param info_in: pad probe information
    :param detector_in: detector
    :param transform_in: image transformation (pre-processing)
    :param device_in: cuda or cpu
    :param detection_threshold_in: detection threshold
    :return:
    """
    global start_time, frames_processed
    start_time = start_time or time.time()

    with nvtx_range("on_frame_probe"):
        buf = info_in.get_buffer()
        print(f"[{buf.pts / Gst.SECOND:6.2f}]")

        with nvtx_range("preprocessing"):
            image_array = buffer_to_numpy(buf, pad_in.get_current_caps())
            image_tensor = transform_in(image_array)
            image_tensor = image_tensor.unsqueeze(0).to(device_in)

        with nvtx_range("inference"):
            with torch.no_grad():
                locs, labels = detector_in(image_tensor)

        with nvtx_range("postprocessing"):
            # Since the decoding is done in the cpu, it is much more efficient to send to complete tensor to cpu, and then
            # decode the results.
            results_per_input = ssd_utils.decode_results((locs.cpu(), labels.cpu()))
            best_results_per_input = [
                ssd_utils.pick_best(results, detection_threshold_in)
                for results in results_per_input
            ]

            for bboxes, classes, scores in best_results_per_input:
                print(f"{bboxes=}")
                print(f"{classes=}")
                print(f"{scores=}")
                print("-------")

        return Gst.PadProbeReturn.OK


def buffer_to_numpy(buf, caps):
    """
    Converts a Gst.Buffer to a numpy array
    :param buf: buffer
    :param caps: object capabilities
    :return: RGB numpy array containing the image
    """
    with nvtx_range("buffer_to_image_tensor"):
        caps_struct = caps.get_structure(0)
        width, height = caps_struct.get_value("width"), caps_struct.get_value("height")

        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        image_array = None

        if is_mapped:
            try:
                image_array = np.ndarray(
                    (height, width, pixel_bytes), dtype=np.uint8, buffer=map_info.data
                )[:, :, :3].copy()
            finally:
                buf.unmap(map_info)

        return image_array


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_file", help="input file path", default="")
    argParser.add_argument(
        "-d",
        "--detection_threshold",
        help="detection threshold",
        default=0.4,
        type=float,
    )
    args = argParser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = (
        torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math="fp32"
        )
        .eval()
        .to(device)
    )

    # Preprocessing
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((300, 300)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229, 0.224, 0.225]),
        ]
    )

    pipeline_definition = f"""
        filesrc location={args.input_file} !
        decodebin !
        nvvideoconvert !
        video/x-raw, format={frame_format} !
        fakesink name=fake_sink
    """

    print("--- PIPELINE DEFINITION ---")
    print(pipeline_definition)
    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_definition)

    # Add probe to fake sink for capturing and processing frames
    pipeline.get_by_name("fake_sink").get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        partial(
            on_frame_probe,
            detector_in=detector,
            transform_in=transform,
            device_in=device,
            detection_threshold_in=args.detection_threshold,
        ),
    )
    pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            msg = pipeline.get_bus().timed_pop_filtered(
                Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            if msg:
                text = msg.get_structure().to_string() if msg.get_structure() else ""
                msg_type = Gst.message_type_get_name(msg.type)
                print(f"{msg.src.name}: [{msg.type}] {text}")
                break
    finally:
        pipeline.set_state(Gst.State.NULL)
