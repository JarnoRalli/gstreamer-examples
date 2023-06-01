# Deepstream Tracking

This example re-implements the example from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-test2/deepstream_test_2.py, hopefully
in a bit more readable format. The example detects and tracks following objects seen in a h264 encoded video stream:

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

## Versions

There are two versions:
* [gst-tracking.py](gst-tracking.py)
  * This version draws bounding box and object information using deepstream's native way.
* [gst-tracking-v2.py](gst-tracking-v2.py)
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

## How to Run the Example

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

