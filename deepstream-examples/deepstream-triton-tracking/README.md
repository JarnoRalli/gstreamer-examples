# 1 Deepstream Tracking with Triton Inferenfce Server

This directory contains several different implementations related to using Triton Inference Server for inference in a Deepstream pipeline.
The programs detect the following objects:

* PGIE_CLASS_ID_VEHICLE = 0
* PGIE_CLASS_ID_BICYCLE = 1
* PGIE_CLASS_ID_PERSON = 2
* PGIE_CLASS_ID_ROADSIGN = 3

Following inference and tracker components are used:

* Primary inference: 4-class detector
* Secondary inference 1: vehicle color classifier
* Secondary inference 2: vehicle make classifier
* Secondary inference 3: vehicle type classifier
* Tracker
  * Configuration file: [dstest2_tracker_config.txt](dstest2_tracker_config.txt)

## 1.1 Versions

* [gst-triton-tracking.py](gst-triton-tracking.py)
  * This version draws bounding box and object information using slightly modified version of the original function.
  * Triton and the Deepstream pipeline are run in the same host.
  * Uses configuration files from `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton`
  * Input video is expected to be h264 encoded
* [gst-triton-tracking-v2.py](gst-triton-tracking-v2.py)
  * This version draws the information so that bounding- and text boxes for smaller objects are drawn first.
  Everything else being the same, smaller objects tend to be further away from the camera. Also bounding bbox colors are different for each object type.
  * Triton and Deepstream pipeline are run in different hosts, gRPC is used for comms.
  * Uses configuration files from `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton-grpc/`
  * Input video is expected to be h264 encoded
* [gst-triton-parallel-tracking-v1.py](./gst-triton-parallel-tracking-v1.py)
  * Allows to generate several (separate) pipelines that all process the same input file. This is done for testing purposes.
  * Triton and the Deepstream pipeline are run in the same host.
  * Uses configuration files from `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/`
  * Input video is expected to be h264 encoded
* [gst-triton-parallel-tracking-v2.py](./gst-triton-parallel-tracking-v2.py)
  * Creates a pipeline that allows to process several video files.
  * Triton and the Deepstream pipeline are run in the same host.
  * Uses configuration files from `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/`
  * Uses tiling
  * Input videos are expected to be h264 encoded

## 1.2 Observations

### 1.2.1 DeepstreamSDK 6.1.1

When using the `nvv4l2h264enc` encoder in the file-sink branch the pipeline became unresponsive after having processed some frames. It seems to work with `x264enc`
without any problems.

### 1.2.2 DeepstreamSDK 6.3

The Gst plug-in `x264enc` apparently has been removed.

## 1.3 Requirements

* DeepStreamSDK 6.1.1 or 6.3
* Nvidia Container toolkit and Docker compose (if using Docker)
* Python >=3.6
* Gst-python
* pyds 1.1.5
* gstreamer1.0-plugins-good
* gstreamer1.0-plugins-bad
* gstreamer1.0-plugins-ugly
* Triton Inference Server (locally built or Docker image)

## 1.4 How to Run the Example Locally

Since the `gst-triton-tracking-v2.py` uses configuration files from `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton-grpc`,
the expectation is that the Triton server is running in the same machine as where the code `gst-triton-tracking-v2.py` is run from. If this is not the case,
then you need to modify the IP-address of the Triton server in the configuration files.

### 1.4.1 Building Models

The first step is to build the TensorRT models:

```bash
/opt/nvidia/deepstream/deepstream/samples/prepare_ds_triton_model_repo.sh
```

Then you start the Triton server:

```bash
tritonserver \
    --log-verbose=2 --log-info=1 --log-warning=1 --log-error=1 \
    --model-repository=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
```

### 1.4.2 Running the Tracking Example

To get help regarding input parameters, execute:

```bash
python3 gst-triton-tracking-v2.py -h
```

In order to process an mp4 file (with h264 encoded video), execute the following:

```bash
python3 gst-triton-tracking-v2.py -i <PATH-TO-INPUT-FILE> -o <PATH-TO-OUTPUT-FILE>
```

If you have DeepStream with samples installed, you can execute the following:

```bash
python3 gst-triton-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

## 1.5 How to Run the Example Using Docker

Here the expectation is that both the Docker container running the Triton server and the container where
the `gst-triton-tracking-v2.py` code is executed from, are running in the same host. We need to modify
the address of the Triton server from `localhost` to `triton-server` in the configuration files, as is explained later on.
You need to build the `deepstream-6.3` Docker image first.

```bash
cd gstreamer-examples/docker
docker build -t deepstream-6.3 -f ./Dockerfile-deepstream-6.3-triton-devel .
```

### 1.5.1 Launch Triton Server Using Docker Compose

Launch Triton server using Docker compose as follows:

```bash
docker compose up --build
```

Verify that Triton is running correctly by executing the following command in the host computer:

```bash
curl -v http://localhost:8000/v2/health/ready
```

If Triton is running correctly, you should get an answer similar to:

```bash
*   Trying 127.0.0.1:8000...
* Connected to localhost (127.0.0.1) port 8000 (#0)
> GET /v2/health/ready HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.81.0
> Accept: */*
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
<
* Connection #0 to host localhost left intact
```

### 1.5.2 Launch gst-triton-tracking-v2.py

Next we launch the Docker container that we use for executing the tracking code. Following commands
are run in the host. First we enable any client to interact with the local X server:

```bash
xhost +
```

Then we start the Docker container by running:

```bash
docker run -i -t \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/gstreamer-examples \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --network deepstream-triton-tracking_triton-network \
  --gpus all deepstream-6.3 bash
```

Once logged into the container, first we check that the container can display to the host's X server by running
the following:

```bash
glmark2
```

If everything works fine, we should see a video of a rotating horse. Then we verify that we can access the Triton
server:

```bash
curl -v http://triton-server:8000/v2/health/ready
```

The last step before executing the code is to replace the address `localhost:` in the configuration files with
`triton-server:` by executing:

```bash
cd /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton-grpc
find . -type f -name "*.txt" -exec sed -i 's/localhost:/triton-server:/g' {} +
find . -type f -name "*.txt" -exec sed -i 's/enable_cuda_buffer_sharing: true/enable_cuda_buffer_sharing: false/g' {} +
```

Now we are ready to run the Triton-tracking code.

```bash
cd /home/gstreamer-examples
python3 gst-triton-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

### 1.5.3 Launch gst-triton-parallel-tracking-v1.py

We use the same Docker container where the Triton server is running. First find out the container ID:

```bash
docker ps
docker exec -i -t <ID> bash
```

Use the ID that corresponds to the `deepstream-triton-tracking-triton-server` container.

```bash
cd /home/gstreamer_examples
python3 gst-triton-parallel-tracking-v1.py -n 2 -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

### 1.5.4 Launch gst-triton-parallel-tracking-v2.py

We use the same Docker container where the Triton server is running. First find out the container ID:

```bash
docker ps
docker exec -i -t <ID> bash
```

Use the ID that corresponds to the `deepstream-triton-tracking-triton-server` container.

```bash
cd /home/gstreamer_examples
python3 gst-triton-parallel-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```
### 1.5.5 Test Pipelines

Following are test pipelines that can be launched with `gst-launch-1.0`.

**Single input stream**

```bash
gst-launch-1.0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! \
m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 ! nvinferserver config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt ! \
nvtracker tracker-width=640 tracker-height=480 ll-config-file=config_tracker_NvDCF_perf.yml \
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! nvdsosd display-clock=1 ! \
nvvideoconvert ! nveglglessink
```

**4 Input streams with video- and filesinks**

This pipeline connects 4 `nvurisrcbin` to a single image processing pipeline.

```bash
gst-launch-1.0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_1 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_2 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_3 \
nvstreammux name=m width=1280 height=720 batch-size=4 ! \
nvinferserver config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt ! \
nvtracker tracker-width=640 tracker-height=480 ll-config-file=config_tracker_NvDCF_perf.yml \
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! \
nvmultistreamtiler rows=2 columns=2 width=1280 height=720 ! \
nvdsosd ! tee name=t t. ! queue ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nvv4l2h264enc profile=High bitrate=10000000 ! h264parse ! matroskamux ! \
filesink location=triton_4_stream_output.mkv t. ! queue ! nvvideoconvert ! nveglglessink
```

**20 Input streams with video- and filesinks**

This pipeline connects 20 `nvurisrcbin` to a single image processing pipeline.

```bash
gst-launch-1.0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_1 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_2 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_3 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_4 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_5 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_6 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_7 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_8 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_9 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_10 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_11 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_12 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_13 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_14 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_15 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_16 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_17 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! queue ! m.sink_18 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 ! queue ! m.sink_19 \
nvstreammux name=m width=1280 height=720 batch-size=20 ! \
nvinferserver config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt ! queue ! \
nvtracker tracker-width=640 tracker-height=480 ll-config-file=config_tracker_NvDCF_perf.yml \
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! queue ! \
nvmultistreamtiler rows=5 columns=4 width=1280 height=720 ! queue ! \
nvdsosd ! tee name=t t. ! queue ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nvv4l2h264enc profile=High bitrate=10000000 ! h264parse ! matroskamux ! \
filesink location=triton_20_stream_output.mkv t. ! queue ! nvvideoconvert ! nveglglessink
```

