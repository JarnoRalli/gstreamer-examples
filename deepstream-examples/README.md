# DEEPSTREAM EXAMPLES

This directory contains Deepstream related examples. Example code, along with configuration files etc., are placed inside sub-directories.

# Examples

List of examples:

* [deepstream-tracking](deepstream-tracking/README.md)
  * 4-class object detector with tracking
* [deepstream-triton-tracking](deepstream-triton-tracking/README.md)
  * 4-class object detector with tracking, uses local version of the Triton Inference Server for inference

Before executing any of the tests, make sure that you have installed all the required components.

# Installing DeepStream SDK

Follow these instructions for installing the DeepStream SDK 
[https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html).

After installation, verify that `nvinfer` plug-in can be found

```
gst-inspect-1.0 nvinfer
```

If it's not found, you might have to execute the following script:

```
sudo /opt/nvidia/deepstream/deepstream/install.sh
```

# Installing DeepStream Python Bindings

Information regarding DeepStream Python bindings can be found from here [https://github.com/NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).
You can download ready to install packages from here [https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases).
After downloading the corresponding wheel-package, you can install it by executing the following from the same directory where the package was downloaded:

```
pip3 install pyds-1.1.4-py3-none-linux_x86_64.whl
```

Replace `pyds-1.1.4-py3-none-linux_x86_64.whl` with the version that you downloaded.


# Installing Triton Inference Server

Before executing those examples that use Triton, you first need to install it locally. First install the following package(s):

```
sudo apt-get install libnccl2
```

Then clone the Triton repository and build it:

```
git clone git@github.com:triton-inference-server/server.git
cd server
git checkout tags/v2.29.0
python3 build.py \
    --enable-stats --enable-logging --enable-tracing --enable-gpu \
    --backend tensorrt --backend onnxruntime --backend python --backend pytorch --backend tensorflow2
```

Copy recently built backends where Triton can find them:

```
cd build/install
sudo cp -vr ./backends /opt/tritonserver
```

## Environment Variables

Triton libraries need to be discoverable by the the dynamic library loader:

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<TRITON-SOURCE>/build/install/lib
```

Replace `<TRITON-SOURCE>` with the location where Triton source code was cloned.

By default `nvinferserver` Gst plug-in is not discoverable, so we need to add the following:

```
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
```

Make sure that `trtexec` is found:

```
which trtexec
```

If it cannot be found, but it is installed, you can add it to path:

```
export PATH=${PATH}:/usr/src/tensorrt/bin/
```

## Build the Model Repo

We will use the models shipped with the DeepStream SDK. However, first make sure that `trtexec` is found:

```
trtexec --version
```

Build the models shipped with DeepStream SDK

```
cd /opt/nvidia/deepstream/deepstream/samples
./prepare_ds_triton_model_repo.sh
```


## Testing Triton Installation

Test that the `nvinferenceserver` plugin can be found

```
gst-inspect-1.0 nvinferserver
```

Test that the model repo can be loaded:

```
cd <TRITON-SOURCE>/build/install/bin
./tritonserver \
    --log-verbose=2 --log-info=1 --log-warning=1 --log-error=1 \
    --model-repository=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
```

Replace `<TRITON-SOURCE>` with the location where Triton source code was cloned.

