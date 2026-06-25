# Environment Instructions

When executing the following scripts, use the specified Docker containers. If the container is not running, the agent should initialize it from the existing images.

## Container Mapping

* **`gst-yolox-bytetrack-cpudec.py`**:
  * Dockerfile: `../docker/Dockerfile-gstreamer-1.28`
  * Image: `gstreamer-1.28`
  * Container Name: `gstreamer-1.28-env`
* **`gst-yolox-bytetrack-gpudec.py`**:
  * Dockerfile: `../docker/Dockerfile-gstreamer-1.28-cuda`
  * Image: `gstreamer-1.28-cuda`
  * Container Name: `gstreamer-1.28-cuda-env`
* **`gst-bytetrack.py`**:
  * Dockerfile: `../docker/Dockerfile-gstreamer-1.28-cuda`
  * Image: `gstreamer-1.28-cuda`
  * Container Name: `gstreamer-1.28-cuda-env`

## Execution Protocol

1. Check if the required `Container Name` for the script is running using `docker ps`.
2. If the container exists but is stopped, run `docker start <Container Name>`. If it does not exist, initialize it using the appropriate command:
   * **For CPU (`gstreamer-1.28-env`):**
     ```bash
     docker run -d --rm \
       --name gstreamer-1.28-env \
       -v $(pwd):/workspace \
       -w /workspace \
       gstreamer-1.28 tail -f /dev/null
     ```
   * **For GPU (`gstreamer-1.28-cuda-env`):**
     ```bash
     docker run -d --rm \
       --name gstreamer-1.28-cuda-env \
       --gpus all \
       -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics \
       -v $(pwd):/workspace \
       -w /workspace \
       gstreamer-1.28-cuda tail -f /dev/null
     ```
3. Ensure the script is executed within the container's shell environment using:
   `docker exec <Container Name> python3 <script_name>`

## Gstreamer MCP

If GStreamer MCP (Model Context Protocol) has been configured, then make sure that the corresponding Docker container is running as per this
Docker compose file `docker/docker-compose.yml`. When possible, use the tools defined in the MCP for solving GStreamer related problems.

