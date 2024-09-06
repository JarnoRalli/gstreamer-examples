# Docker Images

This directory contains docker files used for generating docker containers where the examples can be run.

* [Dockerfile-deepstream](Dockerfile-deepstream)
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

# 1 Creating Docker Images

Following sections show how to:

* Install Docker
* Build the Docker images

## 1.1 Installing Docker

Before creating the docker image, you need to install Nvidia's Container Toolkit. Instructions can be found here:

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
docker build -t nvidia-deepstream-samples -f ./Dockerfile-deepstream .
