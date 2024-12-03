# Conda Virtual Environments

If you use conda for managing Python virtual environments, first you need to install either [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

# 1 Libmamba Solver

Conda's own solver is very slow, so I recommend using `Libmamba`. To use the new solver, first update conda in the base environment (optional step):

```bash
conda update -n base conda
```

Then install and activate `Libmamba` as the solver:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

# 2 Environments

Following YAML configuration files for Conda environments are available:

* [gst-pytorch-gpu-python3.8.yml](./gst-pytorch-gpu-python3.8.yml)
  * **Environment name:** gst-pytorch-gpu-python3.8
  * **Contains:** python 3.8, pytorch, pytorch-cuda=11.6, gstreamer, matplotlib, numpy, etc.
* [gst-pytorch-gpu-python3.10.yml](./gst-pytorch-gpu-python3.10.yml)
  * **Environment name:** gst-pytorch-gpu-python3.10
  * **Contains:** python 3.10, pytorch, pytorch-cuda=12.1, gstreamer, matplotlib, numpy, etc.
* [caffe.yml](./caffe.yml)
  * **Environment name:** caffe
  * **Contains:** python 3.7, caffe, opencv, pillow, etc.

You can create a new virtual environment as follows:

```bash
conda env create -f <NAME-OF-THE-FILE>
```

Once the environment has been created, you can activate it by executing the following command:

```bash
conda activate <NAME-OF-THE-ENVIRONMENT>
```

[!WARNING]
If you use Conda environment, in some cases you have to make sure that the libraries from the Conda environment are preferred over the system libraries.
This is done by setting the `LD_LIBRARY_PATH` variable. First activate the Conda environment that contains all the required libraries, and then run

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

