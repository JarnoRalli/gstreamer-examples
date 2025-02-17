# 1 RTSP Server

This directory implements RTSP server that can be used for playing back video files as RTSP streams. Contents are as follows:

* [Dockerfile](./Dockerfile)
  * Docker container used for running the RTSP server
* [docker-compose.yml](./docker-compose.yml)
  * Docker compose file that can be used for starting the service. By default
  video files from `$HOME/Videos` are used. You need to copy the files
  to that folder and and the modify the docker compose file so that the correct files
  are played back
* [rtsp-server.py](./rtsp-server.py)
  * The file that starts the RTSP server

# 2 Usage

## 2.1 Starting the RTSP Server

Copy the video files that you want to playback to the `$HOME/Videos` folder and modify the file `./docker-compose.yml` so
that the correct files are being played back. Each file will be mapped to RTSP streams, starting from `camera1`.
For example, if we have the following in the `./docker-compose.yml`

```bash
command: python3 /home/gst-rtsp-server.py --files file:///Videos/video1.mp4 file:///Videos/video2.mp4
```

These will be mapped so that:

* `file:///Videos/video1.mp4` --> `rtsp://localhost:8554/camera1`
* `file:///Videos/video2.mp4` --> `rtsp://localhost:8554/camera2`

Once the `./docker-compose.yml` has been modified, you can start the service with

```bash
docker compose up --build
```

## 2.2 Playback using ffplay

You can playback the stream, for example, using `ffplay` as follows:

```bash
ffplay rtsp://localhost:8554/camera1
```

## 2.3 Playback Using GStreamer

You can display the RTSP stream using only GStreamer plug-ins by running the following from the host machine:

```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/camera1 protocols=tcp latency=200 ! \
rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! timeoverlay ! autovideosink
```

### 2.4 Playback Using Deepstream in X86/X64

You can display the RTSP stream using GStreamer and Deepstream by running the following from a non-Jetson device:

```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/camera1 protocols=tcp latency=500 ! \
rtph264depay ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! queue ! \
mux.sink_1 nvstreammux name=mux width=1920 height=1080 batch-size=1 live-source=1 ! \
queue ! nvvideoconvert ! queue ! nvdsosd ! queue ! nveglglessink
```

### 2.5 Playback Using Deepstream in Jetson

You can display the RTSP stream using GStreamer and Deepstream by running the following from a Jetson device:

```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/camera1 protocols=tcp latency=500 ! \
rtph264depay ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! queue ! \
mux.sink_1 nvstreammux name=mux width=1920 height=1080 batch-size=1 live-source=1 ! \
queue ! nvvideoconvert ! queue ! nvdsosd ! queue ! nvegltransform ! nveglglessink
```

