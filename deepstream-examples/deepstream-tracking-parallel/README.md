# 1 Deepstream Parallel Tracking

This example shows to split an input stream into two, using a tee-element, so that two different image processing pipelines can process the same stream.
This example processes the split streams using the same inference elements, but they can be different for each stream. It appears that you need to add an
nvstreammux-element into both of the processing streams, after the tee-element, in order for the nvtracker-element to work properly.

* PGIE_CLASS_ID_VEHICLE = 0
* PGIE_CLASS_ID_BICYCLE = 1
* PGIE_CLASS_ID_PERSON = 2
* PGIE_CLASS_ID_ROADSIGN = 3

# 2 Pipeline

Pipeline description.

![Image of the pipeline](./gst-tracking-parallel.pdf)

# 3 Processing Pipeline Configurations

## 3.1 Pipeline 1

Configuration files for the inference- and tracker elements:

* Primary inference: 4-class detector
  * Configuration file: [pgie_config_1.txt](pgie_config_1.txt)
* Secondary inference 1: vehicle color classifier
  * Configuration file: [sgie1_config_1.txt](sgie1_config_1.txt)
* Secondary inference 2: vehicle make classifier
  * Configuration file: [sgie2_config_1.txt](sgie2_config_1.txt)
* Secondary inference 3: vehicle type classifier
  * Configuration file: [sgie3_config_1.txt](gie3_config_1.txt)
* Tracker
  * Configuration file: [tracker_config_1.txt](tracker_config_1.txt)

## 3.2 Pipeline 2

Configuration files for the inference- and tracker elements:

* Primary inference: 4-class detector
  * Configuration file: [pgie_config_2.txt](pgie_config_2.txt)
* Secondary inference 1: vehicle color classifier
  * Configuration file: [sgie1_config_2.txt](sgie1_config_2.txt)
* Secondary inference 2: vehicle make classifier
  * Configuration file: [sgie2_config_2.txt](sgie2_config_2.txt)
* Secondary inference 3: vehicle type classifier
  * Configuration file: [sgie3_config_2.txt](gie3_config_2.txt)
* Tracker
  * Configuration file: [tracker_config_2.txt](tracker_config_2.txt)

## 3.3 Requirements

* DeepStreamSDK 6.1.1
* Python 3.8
* Gst-python
* pyds 1.1.5
* gstreamer1.0-plugins-good
* gstreamer1.0-plugins-bad
* gstreamer1.0-plugins-ugly

## 3.4 How to Run the Example

In order to get help regarding input parameters, execute the following:

```bash
python3 gst-tracking-parallel.py -h
```

In order to process an mp4 file (with h264 encoded video), execute the following:

```bash
python3 gst-tracking-parallel.py -i <PATH-TO-INPUT-FILE>
```

If you have DeepStream with samples installed, you can execute the following:

```bash
python3 gst-tracking-parallel.py -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

You can generate a Graphviz dot file of the pipeline by adding the switch `-d` when launching the example.
The dot file can be converted into pdf as follows:

```bash
dot -Tpdf <NAME-OF-THE-DOT-FILE> -o output.pdf
```

