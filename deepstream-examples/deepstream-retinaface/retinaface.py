"""
This example shows how to capture frames form a Gst-pipeline and process them with PyTorch.

For help regarding the command line arguments, run:
python retinaface.py --help
"""

import sys
import gi
import numpy as np
import argparse
import contextlib
import time
from functools import partial

gi.require_version('Gst', '1.0')
from gi.repository import Gst

#@contextlib.contextmanager
#def nvtx_range(msg):
#    depth = torch.cuda.nvtx.range_push(msg)
#    try:
#        yield depth
#    finally:
#        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input_file", help="input file path", default="")
    argParser.add_argument("--width", help="video width", default=1920)
    argParser.add_argument("--height", help="video height", default=1080)
    args = argParser.parse_args()

    if args.input_file == "":
        sys.exit("No input file has been given!")

    pipeline_definition = f'''
        filesrc location={args.input_file} !
        qtdemux !
        queue !
        h264parse !
        nvv4l2decoder !
        mux.sink_0 nvstreammux width={args.width} height={args.height} batch_size=1 name=mux !
        nvinfer config-file-path=config_retinaface.txt !
        nvvideoconvert !
        nvdsosd !
        queue !
        nveglglessink'''

    print("--- PIPELINE DEFINITION ---")
    print(pipeline_definition)
    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_definition)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            msg = pipeline.get_bus().timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if msg:
                text = msg.get_structure().to_string() if msg.get_structure() else ''
                msg_type = Gst.message_type_get_name(msg.type)
                print(f"{msg.src.name}: [{msg.type}] {text}")
                break
    finally:
        pipeline.set_state(Gst.State.NULL)
