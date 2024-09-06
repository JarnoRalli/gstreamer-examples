# GST-EXAMPLES

These are Gstreamer related examples. Before running the examples, it is a good idea to refresh the GStreamer plugin cache by running the following:

```bash
gst-inspect-1.0
```

# 1 Requirements

* Python 3.8
* Gst-python
* gstreamer1.0-plugins-bad
* gstreamer1.0-libav
* The following are needed only for PyTorch related examples
  * torch
  * torchvision

# 2 Examples

* [gst-qtdemux-h264.py](gst-qtdemux-h264.py)
  * Plays back h264 encoded video stream from a file (e.g. mp4).
* [gst-qtdemux-h264-avdec_aac.py](gst-qtdemux-h264-avdec_aac.py)
  * Plays back h264 encoded video stream and MPEG-4 AAC encoded audio stream from a file (e.g. mp4).
* [gst-pytorch-example-1.py](gst-pytorch-example-1.py)
  * Captures frames from a GStreamer pipeline and passes those to a SSD-detector.
  * Uses [nvtx](https://docs.nvidia.com/nvtx/index.html) to mark sections of the code so that Nsight-systems can be used
  for analyzing the time spent in pre-processing, inference and post-processing stages.
* [gst-pytorch-example-1.1.py](gst-pytorch-example-1.py)
  * Same as above, but post-processing is done by first transferring the `locs` and `labels` tensors
  from gpu- to cpu-memory, and then applying post-processing. Otherwise the tensors are fetched element-wise, making
  the memory transfers highly inefficient.
