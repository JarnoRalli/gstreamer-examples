# Deepstream Tracking with Triton Inferenfce Server

This example re-implements the example from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-test2/deepstream_test_2.py, using Triton
Inference Server. The example detects and tracks following objects seen in a h264 encoded video stream:

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

There are two versions:
* [gst-triton-tracking.py](gst-triton-tracking.py)
  * This version draws bounding box and object information using deepstream's native way.
* [gst-triton-tracking-v2.py](gst-triton-tracking-v2.py)
  * This version draws the information so that bounding- and text boxes for smaller objects are drawn first. 
  Everything else being the same, smaller objects tend to be further away from the camera. Also bounding bbox colors are different for each object type.

## Requirements

* DeepStreamSDK 6.1.1
* Python 3.8
* Gst-python
* pyds 1.1.5
* gstreamer1.0-plugins-good
* gstreamer1.0-plugins-bad
* gstreamer1.0-plugins-ugly
* Triton Inference Server (locally built)

## Build Triton Inference Server

This example uses a locally built version of Triton. The first step is to build Triton Inference Server v2.29.0:

```
git clone git@github.com:triton-inference-server/server.git
cd server
git checkout tags/v2.29.0
python build.py --enable-logging --enable-tracing --enable-gpu
```

Copy recently built backends where Triton can find them:

```
cd build/install
sudo cp -vr ./backends /opt/tritonserver
```

Build the models shipped with DeepStream

```
cd /opt/nvidia/deepstream/deepstream/samples
./prepare_ds_triton_model_repo.sh
```

## Environment Variables

Following environment variables need to be exported:

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<TRITON-SOURCE>/build/install/lib
```

Replace `<TRITON-SOURCE>` with the location where Triton source code was cloned.

```
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
```

Test that the `nvinferenceserver` plugin can be found

```
gst-inspect-1.0 nvinferserver
```

## How to Run the Example

First you need to launch the Tritonserver:

```
cd <TRITON-SOURCE>/build/install/bin
./tritonserver \
    --log-verbose=2 --log-info=1 --log-warning=1 --log-error=1 \
    --model-repository=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
```

Replace `<TRITON-SOURCE>` with the location where Triton source code was cloned.

In order to get help regarding input parameters, execute the following:

`python3 gst-triton-tracking.py -h`

In order to process an mp4 file (with h264 encoded video), execute the following:

`python3 gst-triton-tracking.py -i <PATH-TO-INPUT-FILE> -o <PATH-TO-OUTPUT-FILE>`

Output video, with overlays, is generated.
