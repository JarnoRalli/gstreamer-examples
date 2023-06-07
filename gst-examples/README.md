# GST-EXAMPLES

These are non-deepstream related examples.

## Requirements

* Python 3.8
* Gst-python
* gstreamer1.0-plugins-bad (you probably need this)
* torch
* torchvision

## Examples

* [gst-qtdemux-h264.py](gst-qtdemux-h264.py)
  * Plays back h264 encoded video stream from a file (e.g. mp4).
* [gst-qtdemux-h264-avdec_aac.py](gst-qtdemux-h264-avdec_aac.py)
  * Plays back h264 encoded video stream and MPEG-4 AAC encoded audio stream from a file (e.g. mp4).
* [gst-pytorch-example-1.py](gst-pytorch-example-1.py)
  * Captures frames from a GStreamer pipeline and passes those to a SSD-detector.
