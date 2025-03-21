# Creating Gstreamer Elements

The easiest way for creating GStreamer elements is to use the tools available from the monorepo.
Start by cloning the GStreamer monorepo.

```bash
git clone https://github.com/GStreamer/gstreamer.git
```

Some of the following tools use a script called `gst-indent`, so it'a a good idea to copy it to a location
where it can be found

```bash
sudo cp scripts/gst-indent /usr/local/bin/
```

## 1 Creating an Empty Meson Project

If you use Meson as a build system, you can create an empty Meson project using the shell script `gst-project-maker`.
Following creates a directory called `gst-my_element`

```bash
cd gstreamer/subprojects/gst-plugins-bad/tools/
./gst-project-maker my_element
```

## 2 Creating an Empty GStreamer Element

A script called `gst-element-maker` can be used for creating an empty GStreamer element which can inherit from different
base classes. At least the following base classes are available:

* audiodecoder
* audioencoder
* audiofilter
* audiosink
* audiosrc
* baseparse
* basesink
* basesrc
* basetransform
* element
* videodecoder
* videoencoder
* videofilter
* videosink

For example, you can create an empty element based on the `basetransform` as follows:

```bash
cd gstreamer/subprojects/gst-plugins-bad/tools/
./gst-element-maker my_element basetransform
```
