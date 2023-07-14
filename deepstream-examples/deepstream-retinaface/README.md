# RetinaFace Detector

This example shows how to use a RetinaFace based detector in deepstream. Outputs from the RetinaNet are:
* Facial bounding boxes
* Facial landmarks
* Class probability/confidences

This example uses a [custom parser](../src/retinaface_parser/nvdsparse_retinaface.cpp) for parsing the data from RetinaFace network, which needs to be compiled before executing this example. Before running
this example, you need to download RetinaFace ONNX-file to this directory.

## Testing

You can test the detector either using a video or a single image.

### Videos

To process a video (tested in Ubuntu 20.04 x86):

```shell
gst-launch-1.0 filesrc location=<VIDEO-FILE-URL> ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! mux.sink_0 nvstreammux width=1920 height=1080 batch_size=1 name=mux ! nvinfer config-file-path=config_detector.txt ! nvvideoconvert ! nvdsosd ! queue ! nveglglessink
```

Set the width and height to corresponding/desired values.

### Videos with Secondary Inference

If you have built the `nvinfer_rect` plugin, then you can use that instead of the standard `nvinfer`. In order for GStreamer to discover the plug-in, you need to set the GST_PLUGIN_PATH first to the directory where `libnvdsgst_infer_rect.so` is located. Tested in Ubuntu 20.04 x86:

```shell
gst-launch-1.0 filesrc location=<VIDEO-FILE-URL> ! decodebin ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinfer_rect config-file-path=config_detector.txt ! nvtracker tracker-width=640 tracker-height=640 gpu-id=0 ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_NvDCF_perf.yml enable-past-frame=1 enable-batch-process=1 ! nvinfer_rect config-file-path=config_classifier.txt ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! nvdsosd ! queue ! nveglglessink
```

Currently the `nvinfer_rect` writes the cropped images, sent to the secondary inference, to hard drive. Set the width and height to corresponding/desired values.

### Images

To process a single image (tested in Ubuntu 20.04 x86):

```shell
gst-launch-1.0 filesrc location=<IMAGE-FILE-URL> ! decodebin ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinfer config-file-path=config_detector.txt ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! jpegenc ! filesink location = ./file.jpg
```

If the above doesn't work, following should work for jpeg-images (tested in Jetson Xavier NX with Ubuntu 20.04 aarch64):

```shell
gst-launch-1.0 filesrc location=<JPEG-FILE-URL> ! jpegdec ! nvvideoconvert ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinfer config-file-path=config_detector.txt ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! jpegenc ! filesink location = ./file.jpg
```

