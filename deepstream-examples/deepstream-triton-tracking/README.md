# Deepstream Tracking with Triton Inferenfce Server

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

## Versions

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

## Observations

### DeepstreamSDK 6.1.1

When using the `nvv4l2h264enc` encoder in the file-sink branch the pipeline became unresponsive after having processed some frames. It seems to work with `x264enc`
without any problems.

### DeepstreamSDK 6.3

The Gst plug-in `x264enc` apparently has been removed.

## Requirements

* DeepStreamSDK 6.1.1 or 6.3
* Nvidia Container toolkit and Docker compose (if using Docker)
* Python >=3.6
* Gst-python
* pyds 1.1.5
* gstreamer1.0-plugins-good
* gstreamer1.0-plugins-bad
* gstreamer1.0-plugins-ugly
* Triton Inference Server (locally built or Docker image)

## How to Run the Example Locally

Since the `gst-triton-tracking-v2.py` uses configuration files from `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton-grpc`,
the expectation is that the Triton server is running in the same machine as where the code `gst-triton-tracking-v2.py` is run from. If this is not the case,
then you need to modify the IP-address of the Triton server in the configuration files.

### Building Models

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

### Running the Tracking Example

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

## How to Run the Example Using Docker

Here the expectation is that both the Docker container running the Triton server and the container where
the `gst-triton-tracking-v2.py` code is executed from, are running in the same host. We need to modify
the address of the Triton server from `localhost` to `triton-server` in the configuration files, as is explained later on.
You need to build the `deepstream-6.3` Docker image first.

```bash
cd gstreamer-examples/docker
docker build -t deepstream-6.3 -f ./Dockerfile-deepstream-6.3-triton-devel .
```

### Launch Triton Server Using Docker Compose

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

### Launch gst-triton-tracking-v2.py

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

### Launch gst-triton-parallel-tracking-v1.py

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

### Launch gst-triton-parallel-tracking-v2.py

We use the same Docker container where the Triton server is running. First find out the container ID:

```bash
docker ps
docker exec -i -t <ID> bash
```

Use the ID that corresponds to the `deepstream-triton-tracking-triton-server` container.

```bash
cd /home/gstreamer_examples
python3 gst-triton-parallel-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
