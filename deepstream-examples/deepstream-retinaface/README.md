# RetinaFace Detector

This example shows how to use a RetinaFace based detector in deepstream. Outputs from the RetinaNet are:
* Facial bounding boxes
* Facial landmarks
* Class probability/confidences

This example uses a [custom parser](../src/retinaface_parser/nvdsparse_retinaface.cpp) for parsing the data from RetinaFace network, which needs to be compiled before executing this example. Before running this example, you need to download RetinaFace ONNX-file to this directory. You can either download it from [GDrive](https://drive.google.com/file/d/1U2wYjZCgnl-HtKtXGAbST2Cp1NCD3ZQJ/view?usp=drive_link), or generate it yourself using a modified version of the [Pytorch_Retinaface](https://github.com/JarnoRalli/Pytorch_Retinaface/tree/feature/run_onnx_model).

## Testing

You can test the detector either using a video or a single image.

### Videos

To process a video, tested in Ubuntu 20.04 x86:

```shell
gst-launch-1.0 filesrc location=<VIDEO-FILE-URL> ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! mux.sink_0 nvstreammux width=1920 height=1080 batch_size=1 name=mux ! nvinfer config-file-path=config_detector.txt ! nvvideoconvert ! nvdsosd ! queue ! nveglglessink
```

Set the width and height to corresponding/desired values.

### Videos with Secondary Inference

This pipeline has both primary and secondary networks. The primary network is a RetinaFace detector and the secondary network is a classifier network from Deepstream. The purpose
of this example is to show how the detections from the primary network are cropped and sent to the secondary network. A modified version of the `nvinfer` gst-plugin writes the
crops to hard drive as images. You need to build  the `nvinferrect` plugin first as per these [instructions](../README.md#2-source-code), and set the `GST_PLUGIN_PATH` to the directory where `libnvdsgst_inferrect.so` is located. Tested in Ubuntu 20.04 x86:

```shell
gst-launch-1.0 filesrc location=<VIDEO-FILE-URL> ! decodebin ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinferrect config-file-path=config_detector.txt ! nvtracker tracker-width=640 tracker-height=640 gpu-id=0 ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_NvDCF_perf.yml enable-past-frame=1 enable-batch-process=1 ! nvinferrect config-file-path=config_classifier.txt ! nvvideoconvert ! nvdsosd ! queue ! nveglglessink
```

Tested in Jetson Xavier NX with Ubuntu 20.04 aarch64:

```shell
gst-launch-1.0 filesrc location=<VIDEO-FILE-URL> ! decodebin ! queue ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinferrect config-file-path=config_detector.txt ! nvtracker tracker-width=640 tracker-height=640 gpu-id=0 ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_NvDCF_perf.yml enable-past-frame=1 enable-batch-process=1 ! nvinferrect config-file-path=config_classifier.txt ! queue ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! nvegltransform ! nveglglessink
```

Set the width and height to corresponding/desired values.

### Images

To process a single image, tested in Ubuntu 20.04 x86:

```shell
gst-launch-1.0 filesrc location=<IMAGE-FILE-URL> ! decodebin ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinfer config-file-path=config_detector.txt ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! jpegenc ! filesink location = ./file.jpg
```

To process a single image, tested in Jetson Xavier NX with Ubuntu 20.04 aarch64:

```shell
gst-launch-1.0 filesrc location=<JPEG-FILE-URL> ! jpegdec ! nvvideoconvert ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinfer config-file-path=config_detector.txt ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! jpegenc ! filesink location = ./file.jpg
```

