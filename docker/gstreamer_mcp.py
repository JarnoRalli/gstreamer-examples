import os
import urllib.request
import urllib.error
import urllib.parse
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("GStreamer-Server")

# Get Docker GStreamer server URL from environment, defaulting to localhost:8000
SERVER_URL = os.environ.get("GST_DOCS_AGENT_URL", "http://localhost:8000")


def _query_agent(endpoint: str, **params) -> dict | str:
    """Queries the Docker GStreamer Doc-Agent HTTP server, returning parsed payload or error string."""
    query_string = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None}
    )
    url = f"{SERVER_URL}/{endpoint}"
    if query_string:
        url += f"?{query_string}"

    try:
        # 10-second timeout to accommodate full gst-inspect scans and dry-run validations
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "success":
                    return payload
                return f"Error from Doc-Agent: {payload.get('detail', 'Unknown error')}"
            return f"Error: Received HTTP status code {response.status}"
    except urllib.error.HTTPError as e:
        try:
            err_data = json.loads(e.read().decode("utf-8"))
            detail = err_data.get("detail", str(e))
            return f"Error from Doc-Agent: {detail}"
        except Exception:
            return f"Error from Doc-Agent: {e.reason} (HTTP {e.code})"
    except urllib.error.URLError as e:
        return (
            f"Error: Could not connect to the GStreamer Doc-Agent at {SERVER_URL}.\n"
            f"Please ensure your GStreamer Docker container is running and port-forwarding is active (e.g. -p 8000:8000).\n"
            f"Inside the container, run:\n"
            f"  fastapi run /workspace/docker/gstreamer_mcp_server.py --port 8000\n"
            f"Details: {e.reason}"
        )
    except Exception as e:
        return f"Unexpected error querying Doc-Agent: {e}"


@mcp.tool()
def check_backend_status() -> str:
    """
    Checks the connectivity and health of the GStreamer Docker backend service.
    Use this tool to verify if the Docker container is active, the API server is reachable,
    and the GStreamer environment inside the container is fully functional.
    """
    res = _query_agent("status")
    if isinstance(res, str):
        return (
            f"🔴 Connection check failed to GStreamer Doc-Agent at {SERVER_URL}.\n\n"
            f"Details: {res}\n\n"
            f"Troubleshooting & Setup Checklist:\n"
            f"1. Is the Docker container running? Run 'docker ps' to verify.\n"
            f"2. Is port-forwarding mapped correctly? Ensure '-p 8000:8000' is included in your docker run command.\n"
            f"3. Is the FastAPI backend running inside the container? Run inside the container:\n"
            f"   fastapi run /workspace/docker/gstreamer_mcp_server.py --port 8000\n"
            f"4. Is the environment variable GST_DOCS_AGENT_URL configured correctly? (Current: {SERVER_URL})"
        )

    gst_ver = res.get("gst_version", "Unknown")
    element_cnt = res.get("element_count", 0)
    is_healthy = res.get("healthy", False)

    if is_healthy:
        return (
            f"🟢 Connection check successful!\n"
            f"Status: Healthy & Active\n"
            f"GStreamer Version: {gst_ver}\n"
            f"Registered Elements: {element_cnt}\n"
            f"Backend Server URL: {SERVER_URL}"
        )
    else:
        err_detail = res.get("detail", "Unknown background error")
        return (
            f"⚠️ Connected, but report unhealthy!\n"
            f"Details: {err_detail}\n"
            f"GStreamer Version: {gst_ver}\n"
            f"Registered Elements: {element_cnt}\n"
            f"Backend Server URL: {SERVER_URL}"
        )


@mcp.tool()
def get_python_gst_docs(python_class_path: str) -> str:
    """
    Gets the exact Python documentation and method signatures for a GStreamer object,
    filtering out noisy inherited GObject methods.
    Example inputs: 'Gst.Element', 'Gst.Pad', 'GstVideo.VideoDecoder'
    """
    res = _query_agent("docs/python", class_path=python_class_path)
    return res if isinstance(res, str) else res.get("data", "")


@mcp.tool()
def get_c_gst_docs(class_path: str) -> str:
    """
    Gets the exact C function signatures, structs, and pointers for a GStreamer object.
    Example inputs: 'Gst.Element', 'Gst.Pad', 'GstVideo.VideoDecoder'
    """
    res = _query_agent("docs/c", class_path=class_path)
    return res if isinstance(res, str) else res.get("data", "")


@mcp.tool()
def list_gst_elements(filter_text: str = None, category_klass: str = None) -> str:
    """
    Lists all available GStreamer elements currently installed and registered in the container.
    Optionally filter elements by name/description (e.g. 'yolox', 'h264') or by semantic class/category
    (e.g. 'Decoder', 'Encoder', 'Demuxer', 'Source', 'Sink').
    """
    res = _query_agent("elements", query=filter_text, klass=category_klass)
    if isinstance(res, str):
        return res

    elements = res.get("data", [])
    if not elements:
        filters = []
        if filter_text:
            filters.append(f"keyword '{filter_text}'")
        if category_klass:
            filters.append(f"category '{category_klass}'")
        return f"No GStreamer elements found matching: {', '.join(filters)}."

    output = []
    header = "Found " + str(len(elements)) + " elements"
    if filter_text or category_klass:
        header_filters = []
        if filter_text:
            header_filters.append(f"keyword '{filter_text}'")
        if category_klass:
            header_filters.append(f"category '{category_klass}'")
        header += f" matching {', '.join(header_filters)}"
    output.append(header + ":\n")

    for item in elements:
        output.append(
            f" - [{item['plugin']}] {item['element']} ({item['klass']}): {item['description']}"
        )

    return "\n".join(output)


@mcp.tool()
def get_gst_element_details(element_name: str) -> str:
    """
    Gets the full specification of a GStreamer element, including a beautifully structured Markdown summary
    of properties (defaults, ranges, types, flags) and static pad templates (MIME types/caps) alongside the
    detailed raw inspect specifications.
    This is essential for finding out what parameters can be set and what formats are supported.
    Example inputs: 'filesrc', 'jpeg2000parse', 'burn-yoloxinference', 'videoconvertscale'
    """
    res = _query_agent("elements/details", name=element_name)
    if isinstance(res, str):
        return res

    data = res.get("data", {})
    schema = data.get("schema", {})
    raw_text = data.get("raw_text", "")

    md_output = []
    md_output.append(f"# Element: {schema.get('name', element_name)}")
    md_output.append(f"**Classification:** {schema.get('klass', 'N/A')}")
    md_output.append(f"**Description:** {schema.get('description', 'N/A')}")
    md_output.append(f"**Author:** {schema.get('author', 'N/A')}\n")

    # Pad Templates
    md_output.append("## Pad Templates")
    pads = schema.get("pad_templates", [])
    if not pads:
        md_output.append("*No static pad templates found.*")
    for pad in pads:
        md_output.append(f"### {pad['direction'].upper()} Pad")
        md_output.append("```text")
        # Format caps nicely by splitting multiple mime-types onto separate lines
        caps_str = pad["caps"].replace("; ", ";\n")
        md_output.append(caps_str)
        md_output.append("```\n")

    # Properties
    md_output.append("## Properties Schema")
    props = schema.get("properties", [])
    if not props:
        md_output.append("*No configurable properties found.*")
    else:
        md_output.append("| Property Name | Type | Default | Access | Description |")
        md_output.append("| :--- | :--- | :--- | :--- | :--- |")
        for p in props:
            # Construct access string (r/w)
            access = []
            if p["readable"]:
                access.append("R")
            if p["writable"]:
                access.append("W")
            access_str = "/".join(access) if access else "None"

            # Format default value safely
            default_val = p["default"]
            if default_val is None:
                default_str = "None"
            elif isinstance(default_val, str) and default_val == "":
                default_str = '""'
            else:
                default_str = str(default_val)

            md_output.append(
                f"| `{p['name']}` | `{p['type']}` | `{default_str}` | {access_str} | {p['description']} |"
            )

    md_output.append("\n---\n")
    md_output.append("## Raw gst-inspect-1.0 Output")
    md_output.append("```text")
    md_output.append(raw_text)
    md_output.append("```")

    return "\n".join(md_output)


@mcp.tool()
def validate_gst_pipeline(pipeline_string: str) -> str:
    """
    Validates a GStreamer pipeline string by performing a timed dry-run inside the container.
    Captures caps negotiation failures, linking errors, and state transitions, returning a detailed
    diagnostic report of any errors or warnings. Use this to verify pipeline syntax before execution.
    Example input: 'filesrc location=IL_Office_2.mp4 ! h264parse ! avdec_h264 ! fakesink'
    """
    res = _query_agent("pipelines/validate", pipeline=pipeline_string)
    if isinstance(res, str):
        return res
    return res.get("data", {}).get("diagnostic", "No diagnostic available.")


if __name__ == "__main__":
    mcp.run()
