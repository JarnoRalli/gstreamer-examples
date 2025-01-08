from setuptools import setup, find_packages

setup(
    name="helpers",
    version="0.0.1",
    author="Jarno Ralli",
    author_email="jarno@ralli.fi",
    maintainer="Jarno Ralli",
    maintainer_email="jarno@ralli.fi",
    description="Set of tools to make creation of gst pipelines easier",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JarnoRalli/gstreamer-examples",
    license="BSD License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyds>=1.1.1",
    ],
    include_package_data=True,
)
