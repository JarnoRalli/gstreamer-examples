# Docker Images

This directory contains docker files used for generating docker images where the examples can be run.

* [Dockerfile-deepstream-6.3-triton-devel](Dockerfile-deepstream-6.3-triton-devel)
  * Docker container with DeepStream 6.3 plus samples, Triton, and DeepStream Python bindings
  * Based on nvcr.io/nvidia/deepstream:6.3-triton-multiarch
  * glmark2 for testing OpenGL inside the container
  * mesa-utils for glxinfo
  * With nvinferserver (Triton) plug-in
* [Dockerfile-deepstream-6.1.1-devel](Dockerfile-deepstream-6.1.1-devel)
  * Docker container with DeepStream 6.1.1 plus samples and DeepStream Python bindings
  * Based on nvcr.io/nvidia/deepstream:6.1.1-samples
  * glmark2 for testing OpenGL inside the container
  * mesa-utils for glxinfo
* [Dockerfile-deepstream-6.0.1-devel](Dockerfile-deepstream-6.0.1-devel)
  * Docker container with Deepstream 6.0.1 plus samples and Deepstream Python bindings
  * Based on nvcr.io/nvidia/deepstream:6.0.1-samples
  * glmark2 for testing OpenGL inside the container
  * mesa-utils for glxinfo
  * cuda-tookit
  * tensorrt-dev
  * no nvinferserver (Triton) plug-in
* [Dockerfile-rtsp-server](Dockerfile-rtsp-server)
  * Docker container with RTSP GStreamer components
  * Based on ubuntu:20.04
  * gstreamer1.0-plugins-base
  * gstreamer1.0-plugins-good
  * gstreamer1.0-plugins-bad
  * gstreamer1.0-plugins-ugly
  * gstreamer1.0-rtsp

# 1 Creating Docker Images

Following sections show how to:

* Install Docker
* Build the Docker images

## 1.1 Installing Docker

If you want to create a Docker image that uses Nvidia's GPU, you first need to install Nvidia's Container Toolkit.
Instructions can be found here:

* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Once you have installed everything, verify that Nvidia's Container Toolkit is working by executing:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

You should see output following (or similar) output:

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:09:00.0  On |                  N/A |
| 32%   38C    P0    34W / 151W |    735MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## 1.2 Create the Docker Image

After this you can create the docker image used in the examples.

```bash
docker build -t deepstream-6.3 -f ./Dockerfile-deepstream-6.3-triton-devel .
```

## 1.3 Test the Docker Image

Some of the examples use GStreamer plugin `nveglglessink` for showing the results in realtime. `nveglglessink`
depends on OpenGL, so making sure that OpenGL works inside the container is essential. Make sure that `DISPLAY`
environment variable has been set:

```bash
env | grep DISPLAY
```
If it is not set, then you need to set it:

```bash
export DISPLAY=:<DISPLAY_NR>
```

Replace `<DISPLAY_NR>` with the actual display which is typically `0` or `1`.

Then start the container:

```bash
xhost +
docker run -i -t --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/gstreamer-examples \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --gpus all deepstream-6.3 bash
```

Then execute the following inside the container:

```bash
glxinfo | grep OpenGL
```

You should see something similar to:

```bash
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA GeForce GTX 1070/PCIe/SSE2
OpenGL core profile version string: 4.6.0 NVIDIA 525.60.13
OpenGL core profile shading language version string: 4.60 NVIDIA
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile
OpenGL core profile extensions:
OpenGL version string: 4.6.0 NVIDIA 525.60.13
OpenGL shading language version string: 4.60 NVIDIA
OpenGL context flags: (none)
OpenGL profile mask: (none)
OpenGL extensions:
OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 525.60.13
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.20
OpenGL ES profile extensions:
```

If the `OpenGL vendor string` is `NVIDIA Corporation`, execute an OpenGL test application inside the container:

```bash
glmark2
```

A window should pop-up, displaying a horse.

