# RetinaFace Detector

This example shows how to use a RetinaFace based detector in deepstream. Outputs from the RetinaNet are:
* Facial bounding boxes
* Facial landmarks
* Class probability/confidences

This example uses a [custom parser](../src/retinaface_parser/nvdsparse_retinaface.cpp) for parsing the data from RetinaFace network, which needs to be compiled before executing this example. Before running
this example, you need to download RetinaFace ONNX-file to this directory.

## Testing

You can test the network either using a video or a single image. To process a video:

```shell
gst-launch-1.0 filesrc location=<VIDEO-FILE-URL> ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! mux.sink_0 nvstreammux width=1920 height=1080 batch_size=1 name=mux ! nvinfer config-file-path=config_retinaface.txt ! nvvideoconvert ! nvdsosd ! queue ! nveglglessink
```

To process a single image:

```shell
gst-launch-1.0 filesrc location=<IMAGE-FILE-URL> ! decodebin ! mux.sink_0 nvstreammux width=640 height=640 batch_size=1 name=mux ! nvinfer config-file-path=config_retinaface.txt ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! jpegenc ! filesink location = ./file.jpg
```

Set the width and height to corresponding/desired values.
