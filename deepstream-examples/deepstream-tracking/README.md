# 1 Deepstream Tracking

This example re-implements the example from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-test2/deepstream_test_2.py, hopefully
in a bit more readable format. The example detects and tracks following objects seen in a video stream:

* PGIE_CLASS_ID_VEHICLE = 0
* PGIE_CLASS_ID_BICYCLE = 1
* PGIE_CLASS_ID_PERSON = 2
* PGIE_CLASS_ID_ROADSIGN = 3

Following inference and tracker components are used:

* Primary inference: 4-class detector
  * Configuration file: [dstest2_pgie_config.txt](dstest2_pgie_config.txt)
* Secondary inference 1: vehicle color classifier
  * Configuration file: [dstest2_sgie1_config.txt](dstest2_sgie1_config.txt)
* Secondary inference 2: vehicle make classifier
  * Configuration file: [dstest2_sgie2_config.txt](dstest2_sgie2_config.txt)
* Secondary inference 3: vehicle type classifier
  * Configuration file: [dstest2_sgie3_config.txt](dstest2_sgie3_config.txt)
* Tracker
  * Configuration file: [dstest2_tracker_config.txt](dstest2_tracker_config.txt)

## 1.1 Versions

There are two versions:
* [gst-tracking.py](gst-tracking.py)
  * This version draws bounding box and object information using deepstream's native way.
* [gst-tracking-v2.py](gst-tracking-v2.py)
  * This version draws the information so that bounding- and text boxes for smaller objects are drawn first.
  Everything else being the same, smaller objects tend to be further away from the camera. Also bounding bbox colors are different for each object type.

## 1.2 Requirements

* DeepStreamSDK 6.1.1
* Python 3.8
* Gst-python
* pyds 1.1.5
* gstreamer1.0-plugins-good
* gstreamer1.0-plugins-bad
* gstreamer1.0-plugins-ugly

## 1.3 How to Run the Example

In order to get help regarding input parameters, execute the following:

```bash
python3 gst-tracking-v2.py -h
```

In order to process an mp4 file (with h264 encoded video), execute the following:

```bash
python3 gst-tracking-v2.py -i <PATH-TO-INPUT-FILE> -o <PATH-TO-OUTPUT-FILE>
```

If you have DeepStream with samples installed, you can execute the following:

```bash
python3 gst-tracking-v2.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

## 1.4 Test Pipelines

Following are test pipelines that can be launched with `gst-launch-1.0`. Requirements for running the pipelines:

* Deepstream 6.3

**Processing pipeline with videosink**

```bash
gst-launch-1.0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! \
m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 ! nvinfer config-file-path=dstest2_pgie_config.txt ! \
nvtracker tracker-width=640 tracker-height=480 ll-config-file=config_tracker_NvDCF_perf.yml \
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! nvdsosd display-clock=1 ! \
nvvideoconvert ! nveglglessink
```

**Processing pipeline with video- and filesinks**

```bash
gst-launch-1.0 \
nvurisrcbin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! \
m.sink_0 nvstreammux name=m width=1280 height=720 batch-size=1 ! nvinfer config-file-path=dstest2_pgie_config.txt ! \
nvtracker tracker-width=640 tracker-height=480 ll-config-file=config_tracker_NvDCF_perf.yml \
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! nvdsosd display-clock=1 ! \
tee name=t t. ! queue ! nvvideoconvert ! 'video/x-raw(memory:NVMM), format=NV12' ! nvv4l2h264enc ! h264parse ! matroskamux ! \
filesink location=output.mkv t. ! queue ! nvvideoconvert ! nveglglessink
```

