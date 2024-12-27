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

## 2.1 Playback

* [gst-qtdemux-h264.py](gst-qtdemux-h264.py)
  * Plays back h264 encoded video stream from a file (e.g. mp4).
* [gst-qtdemux-h264-avdec_aac.py](gst-qtdemux-h264-avdec_aac.py)
  * Plays back h264 encoded video stream and MPEG-4 AAC encoded audio stream from a file (e.g. mp4).

## 2.2 PyTorch

* [gst-pytorch-example-1.py](gst-pytorch-example-1.py)
  * Captures frames from a GStreamer pipeline and passes those to a SSD-detector.
  * Uses [nvtx](https://docs.nvidia.com/nvtx/index.html) to mark sections of the code so that Nsight-systems can be used
  for analyzing the time spent in pre-processing, inference and post-processing stages.
* [gst-pytorch-example-1.1.py](gst-pytorch-example-1.py)
  * Same as above, but post-processing is done by first transferring the `locs` and `labels` tensors
  from gpu- to cpu-memory, and then applying post-processing. Otherwise the tensors are fetched element-wise, making
  the memory transfers highly inefficient.

## 2.3 RTSP Server

This example launches an RTSP server and streams a video over RTSP, encoded in H264. Easiest way to run the program is to execute it
inside the Docker container [Dockerfile-rtsp-server](../docker/Dockerfile-rtsp-server). For more instructions regarding how to build
the image, take a look at the [README.md](../docker/README.md). Once you have created the Docker image, you can start the container
by running the following command from the directory where the file `gst-rtsp-server.py` is found.

```bash
docker run -p 8554:8554 -v $(pwd):/home -it gst-rtsp-server bash
```

Above command exposes the port 8554 from the container and mounts the directory where the command is run to a directory `/home` inside
the container. You can copy the video that you want to stream to the same directory where the `gst-rtsp-server.py` is found, and
thus make it available to the container. Once inside the container, run the following to start streaming:

```bash
cd /home
python3 gst-rtsp-server.py --file=<PATH-TO-VIDEO-FILE>
```

### 2.3.1 Show the RTSP Stream Using GStreamer

You can display the RTSP stream using only GStreamer plug-ins by running the following from the host machine:

```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/test protocols=tcp latency=200 ! \
rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! timeoverlay ! autovideosink
```

### 2.3.2 Show the RTSP Stream Using Deepstream From X86/X64

You can display the RTSP stream using GStreamer and Deepstream by running the following from a non-Jetson device:

```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/test protocols=tcp latency=500 ! \
rtph264depay ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! queue ! \
mux.sink_1 nvstreammux name=mux width=1920 height=1080 batch-size=1 live-source=1 ! \
queue ! nvvideoconvert ! queue ! nvdsosd ! queue ! nveglglessink
```

### 2.3.2 Show the RTSP Stream Using Deepstream From Jetson

You can display the RTSP stream using GStreamer and Deepstream by running the following from a Jetson device:

```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/test protocols=tcp latency=500 ! \
rtph264depay ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! queue ! \
mux.sink_1 nvstreammux name=mux width=1920 height=1080 batch-size=1 live-source=1 ! \
queue ! nvvideoconvert ! queue ! nvdsosd ! queue ! nvegltransform ! nveglglessink
```