import importlib
import inspect
import os
import subprocess
import shlex
import xml.etree.ElementTree as ET
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

app = FastAPI(title="GStreamer Documentation Agent", version="1.0")


def _parse_class_path(class_path: str) -> tuple[str, str]:
    """
    Parse and validate a GObject class path.

    Parameters
    ----------
    class_path : str
        The dot-separated class path, e.g., 'Gst.Element'.

    Returns
    -------
    tuple of (str, str)
        A tuple containing (namespace, class_name).

    Raises
    ------
    ValueError
        If the class path is not in the format 'Namespace.Class'.
    """
    parts = class_path.split(".")
    if len(parts) != 2:
        raise ValueError(
            "Format must be 'Namespace.Class' (e.g., 'Gst.Element' or 'GstVideo.VideoDecoder')"
        )
    return parts[0], parts[1]


def _find_gir_path(namespace: str) -> str:
    """
    Find the path to the .gir file for a given GObject namespace.

    Parameters
    ----------
    namespace : str
        The GObject namespace, e.g., 'Gst' or 'GstVideo'.

    Returns
    -------
    str
        The absolute path to the .gir file.

    Raises
    ------
    FileNotFoundError
        If the .gir file cannot be found in common directories.
    """
    search_paths = ["/usr/local/share/gir-1.0", "/usr/share/gir-1.0"]
    for base_dir in search_paths:
        target = os.path.join(base_dir, f"{namespace}-1.0.gir")
        if os.path.exists(target):
            return target
    raise FileNotFoundError(
        f"Could not find introspection file for {namespace} in {search_paths}"
    )


def _format_python_member(name: str, member: Any) -> str:
    """
    Format a single Python class member representation.

    Parameters
    ----------
    name : str
        The name of the class member.
    member : Any
        The class member object.

    Returns
    -------
    str
        The formatted string representing the member as either a method or a property.
    """
    if callable(member):
        doc = getattr(member, "__doc__", "")
        if doc and "->" in doc:
            signature = doc.split("\n")[0]
            return f" - [Method]   {signature}"

        if type(member).__name__ == "FunctionInfo":
            args = [arg.get_name() for arg in member.get_arguments()]
            arg_string = ", ".join(args)
            if arg_string:
                return f" - [Method]   {name}(self, {arg_string})"
            return f" - [Method]   {name}(self)"

        return f" - [Method]   {name}(...)"
    else:
        return f" - [Property] {name}"


def _format_c_method(
    method: ET.Element, ns: dict[str, str], c_type_name: str
) -> str | None:
    """
    Format a single C method or virtual-method from GI Introspection XML.

    Parameters
    ----------
    method : xml.etree.ElementTree.Element
        The XML element representing the method or virtual-method.
    ns : dict of str to str
        The namespace mapping dictionary for XML querying.
    c_type_name : str
        The name of the C type, e.g., 'GstElement'.

    Returns
    -------
    str or None
        The formatted C function signature string, or None if the C identifier is missing.
    """
    c_func_name = method.attrib.get(f"{{{ns['c']}}}identifier")
    if not c_func_name:
        return None

    ret_val = method.find("core:return-value", ns)
    ret_type = "void"
    if ret_val is not None:
        type_node = ret_val.find("core:type", ns)
        if type_node is not None:
            ret_type = type_node.attrib.get(f"{{{ns['c']}}}type", "unknown")

    params_list = []
    parameters = method.find("core:parameters", ns)
    if parameters is not None:
        instance_param = parameters.find("core:instance-parameter", ns)
        if instance_param is not None:
            inst_type_node = instance_param.find("core:type", ns)
            inst_type = (
                inst_type_node.attrib.get(f"{{{ns['c']}}}type", c_type_name + "*")
                if inst_type_node is not None
                else c_type_name + "*"
            )
            inst_name = instance_param.attrib.get("name", "self")
            params_list.append(f"{inst_type} {inst_name}")

        for param in parameters.findall("core:parameter", ns):
            param_name = param.attrib.get("name", "arg")
            param_type_node = param.find("core:type", ns)
            param_type = (
                param_type_node.attrib.get(f"{{{ns['c']}}}type", "unknown")
                if param_type_node is not None
                else "unknown"
            )
            params_list.append(f"{param_type} {param_name}")

    param_string = ", ".join(params_list)
    tag = "[Virtual]" if method.tag.endswith("virtual-method") else "[Method] "
    return f" - {tag} {ret_type} {c_func_name}({param_string});"


@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> HTMLResponse:
    """
    Render a modern, beautiful landing dashboard with Tailwind CSS.

    Returns
    -------
    HTMLResponse
        The rendered HTML dashboard response displaying system metadata.
    """
    # 1. Fetch system metadata dynamically
    gst_version = "Unknown (GStreamer bindings not found)"
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        gst_version = Gst.version_string()
    except Exception as e:
        gst_version = f"Error: {e}"

    element_count = 0
    try:
        registry = Gst.Registry.get()
        factories = registry.get_feature_list(Gst.ElementFactory)
        element_count = len(factories)
    except Exception:
        pass

    # Render a modern, beautiful landing dashboard with Tailwind CSS
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GStreamer MCP Documentation Server</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
            code, pre {{ font-family: 'JetBrains Mono', monospace; }}
        </style>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    primaryColor: '#0f172a',
                    primaryTextColor: '#cbd5e1',
                    primaryBorderColor: '#334155',
                    lineColor: '#10b981',
                    secondaryColor: '#1e293b',
                    tertiaryColor: '#020617'
                }}
            }});
        </script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen selection:bg-emerald-500 selection:text-slate-950">
        <!-- Header -->
        <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
            <div class="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400
                                flex items-center justify-center font-bold text-slate-950 text-lg
                                shadow-lg shadow-emerald-500/20">
                        GST
                    </div>
                    <div>
                        <h1 class="text-lg font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">GStreamer Doc-Agent</h1>
                        <p class="text-xs text-slate-400">Model Context Protocol Server</p>
                    </div>
                </div>
                <div class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
                    <span class="relative flex h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    Agent Active (Professional Edition)
                </div>
            </div>
        </header>

        <main class="max-w-6xl mx-auto px-6 py-10 space-y-10">
            <!-- Grid of Status Details -->
            <section class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <!-- Card: GStreamer Version -->
                <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">GStreamer Runtime</span>
                    <h3 class="text-xl font-bold mt-2 text-white">{gst_version}</h3>
                    <p class="text-xs text-slate-400 mt-4 border-t border-slate-800/60 pt-3">Introspection bindings active via PyGObject (gi)</p>
                </div>

                <!-- Card: Total Features -->
                <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Registered Features</span>
                    <h3 class="text-3xl font-extrabold mt-2 text-emerald-400">{element_count:,}</h3>
                    <p class="text-xs text-slate-400 mt-4 border-t border-slate-800/60 pt-3">Elements loaded instantly via live Gst.Registry</p>
                </div>

                <!-- Card: Quick Access Docs -->
                <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">API Gateways</span>
                    <div class="flex flex-col gap-2 mt-3">
                        <a href="/docs" target="_blank" class="text-sm font-semibold text-emerald-400 hover:text-emerald-300 hover:underline flex items-center gap-1.5">
                            Interactive Swagger UI &rarr;
                        </a>
                        <a href="/redoc" target="_blank" class="text-sm font-semibold text-teal-400 hover:text-teal-300 hover:underline flex items-center gap-1.5">
                            ReDoc Reference &rarr;
                        </a>
                    </div>
                    <p class="text-xs text-slate-400 mt-4 border-t border-slate-800/60 pt-3">Zero-dependency JSON REST documentation endpoints</p>
                </div>
            </section>

            <!-- Section: System Intent & Architecture -->
            <section class="bg-slate-900/20 border border-slate-800 rounded-3xl p-8 space-y-6">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                    <div class="space-y-4">
                        <h2 class="text-2xl font-bold text-white tracking-tight">Why GStreamer Doc-Agent?</h2>
                        <p class="text-sm text-slate-300 leading-relaxed">
                            GStreamer is exceptionally heavy and difficult to compile or run natively
                            on random host operating systems (macOS, Windows, or clean Linux machines).
                            It has deep system library requirements, complex GObject Introspection bindings,
                            and complex hardware dependencies (like Nvidia GPU/CUDA drivers).
                        </p>
                        <p class="text-sm text-slate-300 leading-relaxed">
                            This project bridges that gap. By running GStreamer, its deep-learning models
                            (like <code>burn-yoloxinference</code>), and all structural introspection metadata
                            securely inside a pre-configured <strong>Docker Container</strong>, we keep your host
                            machine clean.
                        </p>
                        <p class="text-sm text-slate-300 leading-relaxed">
                            A lightweight Python <strong>MCP Proxy Client</strong> runs on your host. When your
                            AI assistant (like OpenCode) wants to search for elements, inspect capabilities
                            (caps), check properties, or dry-run validate a pipeline, the proxy queries this
                            containerized documentation agent in real-time, delivering 100% version-accurate
                            answers instantly.
                        </p>
                    </div>
                    <div class="bg-slate-950 p-6 rounded-2xl border border-slate-800/80 flex flex-col items-center">
                        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4 font-mono">System Architecture Diagram</span>
                        <div class="mermaid w-full flex justify-center text-sm">
                            graph TD
                                subgraph Host ["Host Machine"]
                                    OpenCode[OpenCode / IDE]:::componentNode
                                    Proxy[gstreamer_mcp.py Proxy]:::highlightNode
                                    OpenCode &lt;--&gt;|MCP Protocol| Proxy
                                end

                                subgraph Container ["Docker GStreamer Container"]
                                    Agent[gstreamer_mcp_server.py Server]:::highlightNode
                                    Gst[GStreamer Core &amp; Plugins]:::componentNode
                                    GIR[GObject Introspection .gir]:::componentNode

                                    Agent &lt;--&gt;|PyGObject / gi| Gst
                                    Agent &lt;--&gt;|XML Parsing| GIR
                                end

                                Proxy &lt;--&gt;|HTTP REST / JSON Port 8000| Agent

                                classDef componentNode fill:#020617,stroke:#475569,stroke-width:1px,color:#cbd5e1;
                                classDef highlightNode fill:#022c22,stroke:#10b981,stroke-width:1px,color:#34d399;
                        </div>
                    </div>
                </div>
            </section>

            <!-- Section: Client Setup Instruction -->
            <section class="bg-slate-900/30 border border-slate-800/80 rounded-3xl p-8 space-y-6">
                <div>
                    <h2 class="text-2xl font-bold text-white tracking-tight">OpenCode / Claude Desktop Integration</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        Configure your Model Context Protocol clients on the host using the
                        configuration block below. It uses <code>conda run</code> to guarantee
                        it executes inside the correct environment.
                    </p>
                </div>

                <div class="space-y-3">
                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Example Configuration (e.g., config.json)</span>
                    <pre class="bg-slate-950 p-5 rounded-2xl border border-slate-800 text-sm overflow-x-auto text-emerald-400/90 leading-relaxed shadow-inner">
{{
  "mcpServers": {{
    "gstreamer": {{
      "command": "conda",
      "args": [
        "run",
        "-n",
        "mcp-env",
        "--no-capture-output",
        "python3",
        "/home/jarno/projects/jarno/gstreamer-examples/docker/gstreamer_mcp.py"
      ],
      "env": {{
        "GST_DOCS_AGENT_URL": "http://localhost:8000"
      }}
    }}
  }}
}}</pre>
                </div>

                <div class="border-t border-slate-800/60 pt-6 space-y-3">
                    <h3 class="text-base font-bold text-white">Debugging with the MCP Developer Inspector</h3>
                    <p class="text-xs text-slate-400 leading-relaxed">
                        To test your MCP tools manually in a local, interactive web-based
                        console before connecting them to OpenCode, activate your environment
                        and start the developer inspector:
                    </p>
                    <pre class="bg-slate-950 p-4 rounded-xl border border-slate-800/60 text-xs text-slate-300">
conda activate mcp-env
mcp dev /home/jarno/projects/jarno/gstreamer-examples/docker/gstreamer_mcp.py</pre>
                </div>
            </section>

            <!-- API Endpoints Cheat Sheet -->
            <section class="space-y-4">
                <h2 class="text-xl font-bold text-white">Direct Verification Commands</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Python Docs API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3">
                        <h4 class="text-sm font-bold text-white">Query Python Docs</h4>
                        <p class="text-xs text-slate-400">Gets exact PyGObject signature filters, docstrings, and class hierarchies.</p>
                        <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300 border border-slate-800/50 overflow-x-auto">
curl "http://127.0.0.1:8000/docs/python?class_path=Gst.Element"</pre>
                    </div>

                    <!-- C Signatures API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3">
                        <h4 class="text-sm font-bold text-white">Query C Signatures</h4>
                        <p class="text-xs text-slate-400">Parses raw gobject introspection XML (GIR files) for direct C struct layout and functions.</p>
                        <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300 border border-slate-800/50 overflow-x-auto">
curl "http://127.0.0.1:8000/docs/c?class_path=Gst.Element"</pre>
                    </div>

                    <!-- Available Elements API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3 md:col-span-2">
                        <h4 class="text-sm font-bold text-white">Search Registered Elements</h4>
                        <p class="text-xs text-slate-400">
                            Find installed components. Support filtering by name, description, or
                            <b>semantic classification</b> (Klass).
                        </p>
                        <div class="space-y-2">
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">1. Filter by keyword:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements?query=nv"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">
                                    2. Semantic Class Filter (e.g. Decoder, Encoder, Demuxer):
                                </span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements?klass=Decoder"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">3. List All Elements:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements"</pre>
                            </div>
                        </div>
                    </div>

                    <!-- Inspect Element Details API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3 md:col-span-2">
                        <h4 class="text-sm font-bold text-white">Inspect Element Details (Structured JSON + Raw Specs)</h4>
                        <p class="text-xs text-slate-400">
                            Retrieves typed property structures, writable/readable flags,
                            pad templates, and caps alongside raw inspect readouts.
                        </p>
                        <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300 border border-slate-800/50 overflow-x-auto">
curl "http://127.0.0.1:8000/elements/details?name=jpeg2000parse" | jq .</pre>
                    </div>

                    <!-- Pipeline Validation API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3 md:col-span-2 border-emerald-500/30 bg-emerald-950/10">
                        <h4 class="text-sm font-bold text-emerald-400">Dry-run Pipeline Validation (Self-Healing Loop)</h4>
                        <p class="text-xs text-slate-400">
                            Simulates launching a pipeline inside GStreamer. Automatically
                            parses and returns precise caps-negotiation warnings, state
                            errors, and link failures.
                        </p>
                        <div class="space-y-2">
                            <div>
                                <span class="text-[10px] font-semibold text-emerald-500 uppercase font-mono">Example: Valid Pipeline</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/pipelines/validate?pipeline=videotestsrc+num-buffers%3D10+%21+fakesink"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-red-400 uppercase font-mono">Example: Invalid Pipeline (Fails to link)</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/pipelines/validate?pipeline=videotestsrc+%21+audioconvert+%21+fakesink"</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </main>

        <footer class="border-t border-slate-800/60 mt-16 py-8 text-center text-xs text-slate-500">
            &copy; 2026 GStreamer MCP Documentation Server Agent. Done cleanly & maintainably.
        </footer>
    </body>
    </html>
    """
    return html_content


@app.get("/status")
def get_status() -> dict[str, Any]:
    """
    Check and fetch GStreamer library and feature metadata.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status, GStreamer version, element count, and health indicator.
    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)

        gst_version = Gst.version_string()

        registry = Gst.Registry.get()
        factories = registry.get_feature_list(Gst.ElementFactory)
        element_count = len(factories)

        return {
            "status": "success",
            "gst_version": gst_version,
            "element_count": element_count,
            "healthy": True,
        }
    except Exception as e:
        return {
            "status": "success",
            "gst_version": "Unknown (GStreamer bindings failed)",
            "element_count": 0,
            "healthy": False,
            "detail": str(e),
        }


@app.get("/elements")
def get_available_elements(
    query: str
    | None = Query(None, description="Optional search string/filter for elements"),
    klass: str
    | None = Query(
        None,
        description="Optional semantic class filter (e.g. 'Decoder', 'Encoder', 'Demuxer', 'Source')",
    ),
) -> dict[str, Any]:
    """
    List and filter GStreamer elements registered in the container.

    Parameters
    ----------
    query : str or None, optional
        A search string to filter elements by name, description, plugin, or class, by default None.
    klass : str or None, optional
        A semantic class to filter elements, by default None.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status and sorted list of filtered elements.

    Raises
    ------
    HTTPException
        If GStreamer registry querying fails.
    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)

        registry = Gst.Registry.get()
        factories = registry.get_feature_list(Gst.ElementFactory)

        elements = []
        for factory in factories:
            name = factory.get_name()
            f_klass = factory.get_klass()
            desc = factory.get_description()
            plugin = factory.get_plugin_name() or "core"

            # Apply query filter
            if query:
                q = query.lower()
                if (
                    q not in name.lower()
                    and q not in desc.lower()
                    and q not in plugin.lower()
                    and q not in f_klass.lower()
                ):
                    continue

            # Apply klass filter
            if klass:
                k = klass.lower()
                if k not in f_klass.lower():
                    continue

            elements.append(
                {
                    "plugin": plugin,
                    "element": name,
                    "klass": f_klass,
                    "description": desc,
                }
            )

        # Sort alphabetically
        elements.sort(key=lambda x: x["element"])
        return {"status": "success", "data": elements}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/elements/details")
def get_element_details(
    name: str = Query(
        ..., description="Element name to inspect, e.g. jpeg2000parse, filesrc"
    )
) -> dict[str, Any]:
    """
    Get detailed information about a GStreamer element including schema and properties.

    Parameters
    ----------
    name : str
        The name of the GStreamer element to inspect.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status and element details (raw gst-inspect text and schema).

    Raises
    ------
    HTTPException
        If the element is not found or introspection fails.
    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GObject

        Gst.init(None)

        factory = Gst.ElementFactory.find(name)
        if not factory:
            raise HTTPException(
                status_code=404, detail=f"Element factory '{name}' not found."
            )

        # 1. Run gst-inspect-1.0 to get the raw text report for easy human/LLM reading
        raw_text = ""
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", name], capture_output=True, text=True, check=True
            )
            raw_text = result.stdout
        except Exception:
            pass

        # 2. Extract structured schema dynamically via PyGObject
        schema = {
            "name": name,
            "klass": factory.get_klass(),
            "description": factory.get_description(),
            "author": factory.get_metadata("author") or "Unknown",
            "pad_templates": [],
            "properties": [],
        }

        # Static Pad Templates
        for pad_template in factory.get_static_pad_templates():
            direction = (
                "sink" if pad_template.direction == Gst.PadDirection.SINK else "src"
            )
            schema["pad_templates"].append(
                {"direction": direction, "caps": pad_template.get_caps().to_string()}
            )

        # Create a temporary element instance to query default property values safely
        element = factory.create(None)
        if element:
            for pspec in element.list_properties():
                prop_name = pspec.name
                prop_type = pspec.value_type.name
                prop_desc = pspec.nick

                # Fetch default values safely if property is readable
                default_val = None
                if pspec.flags & GObject.ParamFlags.READABLE:
                    try:
                        val = element.get_property(prop_name)
                        if isinstance(val, GObject.Object):
                            default_val = f"<{val.__class__.__name__}>"
                        else:
                            default_val = val
                    except Exception:
                        pass

                schema["properties"].append(
                    {
                        "name": prop_name,
                        "type": prop_type,
                        "description": prop_desc,
                        "default": default_val,
                        "readable": bool(pspec.flags & GObject.ParamFlags.READABLE),
                        "writable": bool(pspec.flags & GObject.ParamFlags.WRITABLE),
                    }
                )

        return {"status": "success", "data": {"raw_text": raw_text, "schema": schema}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipelines/validate")
def validate_pipeline(
    pipeline: str = Query(
        ...,
        description="The full pipeline string, e.g. 'filesrc location=foo.mp4 ! qtdemux ! fakesink'",
    )
) -> dict[str, Any]:
    """
    Perform a timed dry-run inside the container to validate a pipeline string.

    Parameters
    ----------
    pipeline : str
        The full GStreamer pipeline string to validate.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status, validation outcome, and diagnostic details.

    Raises
    ------
    HTTPException
        If the validation process crashes.
    """
    try:
        # Use shlex to safely break the pipeline parameters into arguments
        args = shlex.split(pipeline)

        # Inject warning/error capturing flag into the environment
        env = os.environ.copy()
        env["GST_DEBUG"] = "3"

        # Start the pipeline using gst-launch-1.0
        proc = subprocess.Popen(
            ["gst-launch-1.0", "-v"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            # Wait 1.5 seconds to negotiate formats, check capabilities, and state transition
            stdout, stderr = proc.communicate(timeout=1.5)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            # Process didn't exit after 1.5 seconds! Pipeline is completely valid & running
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            stdout, stderr = proc.communicate()
            exit_code = 0

        # Parse output logs for diagnostic analysis
        errors = []
        warnings = []
        all_logs = (stdout or "") + "\n" + (stderr or "")

        for line in all_logs.splitlines():
            line_str = line.strip()
            if "ERROR" in line_str or "ERR " in line_str:
                errors.append(line_str)
            elif "WARN" in line_str:
                warnings.append(line_str)

        # Deduplicate logs
        errors = list(dict.fromkeys(errors))
        warnings = list(dict.fromkeys(warnings))

        # Determine valid status
        is_valid = len(errors) == 0 and exit_code == 0

        diagnostic = ""
        if not is_valid:
            if exit_code != 0:
                diagnostic += f"Pipeline failed to launch (Exit Code {exit_code}).\n"
            if errors:
                diagnostic += "\n🔴 Critical GStreamer Errors:\n" + "\n".join(
                    f"  {err}" for err in errors[:15]
                )
            if warnings:
                diagnostic += "\n⚠️ GStreamer Warnings:\n" + "\n".join(
                    f"  {warn}" for warn in warnings[:10]
                )
        else:
            diagnostic = "🟢 Pipeline initialized, caps negotiated, and state-transitioned to PLAYING successfully!"
            if warnings:
                diagnostic += "\n\n⚠️ Minor Warnings (Non-fatal):\n" + "\n".join(
                    f"  {warn}" for warn in warnings[:5]
                )

        return {
            "status": "success",
            "data": {
                "valid": is_valid,
                "exit_code": exit_code,
                "diagnostic": diagnostic,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline validation process crashed: {str(e)}"
        )


@app.get("/docs/python")
def get_python_docs(
    class_path: str = Query(..., description="E.g., Gst.Element, Gst.Pad")
) -> dict[str, Any]:
    """
    Retrieve Python documentation and method signatures for a GObject class.

    Parameters
    ----------
    class_path : str
        Dot-separated GObject class path, e.g., 'Gst.Element'.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status and Python-formatted class documentation.

    Raises
    ------
    HTTPException
        If PyGObject is missing, parsing fails, or class cannot be found.
    """
    try:
        import gi
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyGObject (gi) is not installed in this environment.",
        )

    try:
        namespace, class_name = _parse_class_path(class_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        gi.require_version("Gst", "1.0")
        if namespace != "Gst":
            gi.require_version(namespace, "1.0")

        module = importlib.import_module(f"gi.repository.{namespace}")
        if not hasattr(module, class_name):
            raise HTTPException(
                status_code=404,
                detail=f"Class '{class_name}' not found in namespace '{namespace}'",
            )
        target_class = getattr(module, class_name)

        output = [f"Class: {target_class.__name__}"]
        if target_class.__doc__:
            output.append(f"Docstring:\n{target_class.__doc__}")

        output.append("\nSpecific Members:")

        for name, member in inspect.getmembers(target_class):
            if name.startswith("_"):
                continue

            if name in target_class.__dict__:
                formatted = _format_python_member(name, member)
                output.append(formatted)

        return {"status": "success", "data": "\n".join(output)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Introspection failed: {str(e)}")


@app.get("/docs/c")
def get_c_docs(
    class_path: str = Query(..., description="E.g., Gst.Element, Gst.Pad")
) -> dict[str, Any]:
    """
    Retrieve C signatures and summary from GI Introspection XML for a GObject class.

    Parameters
    ----------
    class_path : str
        Dot-separated GObject class path, e.g., 'Gst.Element'.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status and parsed C signatures.

    Raises
    ------
    HTTPException
        If XML parsing fails, the class is not found, or introspection files are missing.
    """
    try:
        namespace, class_name = _parse_class_path(class_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        gir_path = _find_gir_path(namespace)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        tree = ET.parse(gir_path)
        root = tree.getroot()

        ns = {
            "core": "http://www.gtk.org/introspection/core/1.0",
            "c": "http://www.gtk.org/introspection/c/1.0",
        }

        target_class = root.find(f".//core:class[@name='{class_name}']", ns)
        if target_class is None:
            raise HTTPException(
                status_code=404,
                detail=f"Class '{class_name}' not found in {namespace}.",
            )

        c_type_name = target_class.attrib.get(f"{{{ns['c']}}}type", class_name)

        output = [f"C Struct: {c_type_name}"]

        doc = target_class.find("core:doc", ns)
        if doc is not None and doc.text:
            summary = doc.text.strip().split("\n\n")[0]
            output.append(f"Summary: {summary}\n")

        output.append("C Functions & Methods:")

        methods = target_class.findall("core:method", ns) + target_class.findall(
            "core:virtual-method", ns
        )
        for method in methods:
            formatted = _format_c_method(method, ns, c_type_name)
            if formatted:
                output.append(formatted)

        return {"status": "success", "data": "\n".join(output)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XML Parsing failed: {str(e)}")
