[![pre-commit](https://github.com/JarnoRalli/gstreamer-examples/actions/workflows/pre-commit.yml/badge.svg?branch=main&event=push)](https://github.com/JarnoRalli/gstreamer-examples/actions/workflows/pre-commit.yml)

# GSTREAMER-EXAMPLES

This repository contains examples related to GStreamer, Deepstream and Hailo. Some of the examples are written in Python
and some of them are written in C/C++.

# 1 Contents

Directories are as follows:

* [helper-package](helper-package/README.md). A package that contains helper functions and classes.
* [deepstream-examples](deepstream-examples/README.md). Deepstream related examples.
* [hailo-examples](hailo-examples/README.md). Hailo related examples.
* [gst-examples](gst-examples/README.md). Gst-examples.
* [docker](docker/README.md). Docker files for generating containers.
* [conda](conda/README.md). Conda virtual environments.

Paul Bridger has excellent tutorials regarding how to speed up inference. For anyone interested in the subject,
I recommend you to take a look at:
* https://paulbridger.com/posts/video-analytics-pytorch-pipeline/
* https://paulbridger.com/posts/video-analytics-pipeline-tuning/

# 2 Helper-Package

Helpers is a Python package that contains some helper routines for creating gst-pipelines. Most of the examples, if not all,
use modules from this package, so it needs to be available to Python. The Docker images in the directory [docker](./docker/README.md) install
this package automatically. If you need to make the package available in other environments, you can install it as follows.
Firs make sure that you have the latest version of the `build` package installed using the following command:

```bash
python3 -m pip install --upgrade build
```

In order to create the `helper` package, run the following command from the directory where the `pyproject.toml` is located:

```bash
cd helper-package
python3 -m build
```

Above command creates a new directory called `dist` where the package can be found. In order to install the created package,
run the following command from the `dist` directory:

```bash
pip3 install ./helpers-0.0.1-py3-none-any.whl
```

Replace `helpers-0.0.1-py3-none-any.whl` with the actual name/path of the whl-file that was created.

## 2.1 Usage

Once you have installed the `helpers` package, you can use is as follows:

```python
from helpers import gsthelpers
```

## 2.2 Python Packages and Modules

For more information regarding Python packagaging etc., take a look at:

* [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
* [https://docs.python.org/3/tutorial/modules.html#packages](https://docs.python.org/3/tutorial/modules.html#packages)
* [https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html](https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html)
