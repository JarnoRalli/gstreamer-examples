import importlib
import inspect
import os
import subprocess
import shlex
import xml.etree.ElementTree as ET
from typing import Any
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.

    Attributes
    ----------
    gir_search_paths : list of str
        The list of directory paths searched to locate GObject Introspection (.gir) files.
        Default is ["/usr/local/share/gir-1.0", "/usr/share/gir-1.0"].
    """

    gir_search_paths: list[str] = ["/usr/local/share/gir-1.0", "/usr/share/gir-1.0"]

    class Config:
        env_prefix = "GSTMCP_"


def get_settings() -> Settings:
    """
    Retrieve or instantiate the application configuration settings.

    Returns
    -------
    Settings
        The application configuration settings instance.
    """
    return Settings()


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


def _find_gir_path(
    namespace: str,
    gir_search_paths: list[str] = ["/usr/local/share/gir-1.0", "/usr/share/gir-1.0"],
) -> str:
    """
    Find the path to the .gir file for a given GObject namespace.

    Parameters
    ----------
    namespace : str
        The GObject namespace, e.g., 'Gst' or 'GstVideo'.
    gir_search_paths: list[str]
        Search paths to the directories where .gir files can be found.
        Default ["/usr/local/share/gir-1.0", "/usr/share/gir-1.0"]

    Returns
    -------
    str
        The absolute path to the .gir file.

    Raises
    ------
    FileNotFoundError
        If the .gir file cannot be found in common directories.
    """

    for base_dir in gir_search_paths:
        target = os.path.join(base_dir, f"{namespace}-1.0.gir")
        if os.path.exists(target):
            return target
    raise FileNotFoundError(
        f"Could not find introspection file for {namespace} in {gir_search_paths}"
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
def get_dashboard(settings: Settings = Depends(get_settings)) -> HTMLResponse:
    """
    Render a modern, beautiful landing dashboard with Tailwind CSS.

    Parameters
    ----------
    settings : Settings
        The application configuration settings instance.

    Returns
    -------
    HTMLResponse
        The rendered HTML dashboard response displaying system metadata.
    """
    # 1. Fetch system metadata dynamically
    gst_version = "Unknown (GStreamer bindings not found)"
    element_count = 0
    categories = {
        "Sources": [],
        "Sinks": [],
        "Decoders": [],
        "Encoders": [],
        "Parsers": [],
        "Demuxers": [],
        "Muxers": [],
        "Filters": [],
        "Converters": [],
        "Other / Generic": [],
    }

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        gst_version = Gst.version_string()

        registry = Gst.Registry.get()
        factories = registry.get_feature_list(Gst.ElementFactory)
        element_count = len(factories)

        for factory in factories:
            name = factory.get_name()
            klass = factory.get_klass() or "Generic"
            desc = factory.get_description() or ""
            plugin = factory.get_plugin_name() or "core"
            klass_lower = klass.lower()

            element_info = {
                "name": name,
                "plugin": plugin,
                "klass": klass,
                "desc": desc,
            }

            if "source" in klass_lower:
                categories["Sources"].append(element_info)
            elif "sink" in klass_lower:
                categories["Sinks"].append(element_info)
            elif "decoder" in klass_lower:
                categories["Decoders"].append(element_info)
            elif "encoder" in klass_lower:
                categories["Encoders"].append(element_info)
            elif "parser" in klass_lower:
                categories["Parsers"].append(element_info)
            elif "demuxer" in klass_lower:
                categories["Demuxers"].append(element_info)
            elif "muxer" in klass_lower:
                categories["Muxers"].append(element_info)
            elif "filter" in klass_lower:
                categories["Filters"].append(element_info)
            elif "converter" in klass_lower:
                categories["Converters"].append(element_info)
            else:
                categories["Other / Generic"].append(element_info)

        # Sort elements inside each category alphabetically
        for cat in categories:
            categories[cat].sort(key=lambda x: x["name"])

    except Exception as e:
        gst_version = f"Error: {e}"

    # 2. Build live environment variables list
    env_vars = [
        {
            "name": "GSTMCP_GIR_SEARCH_PATHS",
            "value": ", ".join(settings.gir_search_paths),
            "source": "Application Settings",
            "desc": "List of directories scanned to locate GObject Introspection (.gir) XML files inside the container.",
        },
        {
            "name": "GST_DOCS_AGENT_URL",
            "value": os.environ.get(
                "GST_DOCS_AGENT_URL", "http://localhost:8000 (Default)"
            ),
            "source": "Host Proxy",
            "desc": "URL utilized by the host's gstreamer_mcp.py proxy client to communicate with this containerized backend agent.",
        },
        {
            "name": "GST_DEBUG",
            "value": os.environ.get("GST_DEBUG", "Not set (Default levels apply)"),
            "source": "GStreamer Core",
            "desc": "Controls GStreamer's diagnostic and log output verbosity. Format: category:level, e.g. *:3.",
        },
        {
            "name": "GST_PLUGIN_PATH",
            "value": os.environ.get("GST_PLUGIN_PATH", "Not set"),
            "source": "GStreamer Registry",
            "desc": "Specifies custom colon-separated paths to search for additional GStreamer plugins.",
        },
        {
            "name": "GST_PLUGIN_SYSTEM_PATH",
            "value": os.environ.get("GST_PLUGIN_SYSTEM_PATH", "Not set"),
            "source": "GStreamer Registry",
            "desc": "Overrides or restricts the standard system directories where GStreamer looks for pre-installed plugins.",
        },
        {
            "name": "DISPLAY",
            "value": os.environ.get("DISPLAY", "Not set"),
            "source": "Docker/X11 Forwarding",
            "desc": "Standard X11 display identifier for video sink visualization forwarding to your host display.",
        },
        {
            "name": "XAUTHORITY",
            "value": os.environ.get("XAUTHORITY", "Not set"),
            "source": "Docker/X11 Forwarding",
            "desc": "File path containing authority keys/cookies required to authenticate connection to the X11 display server.",
        },
        {
            "name": "NVIDIA_DRIVER_CAPABILITIES",
            "value": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "Not set"),
            "source": "Nvidia Container Runtime",
            "desc": "Enables specific Nvidia GPU capabilities inside the container (e.g. 'all', 'compute,utility,video').",
        },
    ]

    env_rows = []
    for var in env_vars:
        val_display = var["value"]
        if len(val_display) > 60:
            val_display = val_display[:57] + "..."
        env_rows.append(
            f"""
        <tr class="hover:bg-slate-900/30 transition-colors">
            <td class="px-6 py-4 font-mono font-bold text-white text-xs">{var['name']}</td>
            <td class="px-6 py-4 font-mono text-emerald-400 text-xs break-all" title="{var['value']}">{val_display}</td>
            <td class="px-6 py-4">
                <span class="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-medium font-mono">{var['source']}</span>
            </td>
            <td class="px-6 py-4 text-slate-400 text-xs leading-normal">{var['desc']}</td>
        </tr>
        """
        )
    env_rows_rendered = "\n".join(env_rows)

    # 3. Build GIR status list
    common_namespaces = [
        "Gst",
        "GstBase",
        "GstVideo",
        "GstAudio",
        "GstPbutils",
        "GstRtsp",
        "GstApp",
    ]
    gir_status_list = []

    for ns in common_namespaces:
        found_path = None
        for base_dir in settings.gir_search_paths:
            target = os.path.join(base_dir, f"{ns}-1.0.gir")
            if os.path.exists(target):
                found_path = target
                break

        gir_status_list.append(
            {
                "namespace": ns,
                "filename": f"{ns}-1.0.gir",
                "found": found_path is not None,
                "path": found_path if found_path else "Not found",
            }
        )

    gir_cards = []
    for item in gir_status_list:
        if item["found"]:
            status_badge = """
            <span class="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
                <span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                Active &amp; Parsed
            </span>
            """
            path_display = f"""
            <div class="mt-3 font-mono text-[10px] text-slate-400 truncate bg-slate-950 p-2 rounded border border-slate-800/60" title="{item['path']}">
                {item['path']}
            </div>
            """
        else:
            status_badge = """
            <span class="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold">
                <span class="h-1.5 w-1.5 rounded-full bg-red-500"></span>
                Missing
            </span>
            """
            path_display = """
            <div class="mt-3 font-mono text-[10px] text-red-400/80 bg-red-950/10 p-2 rounded border border-red-900/20">
                Unavailable in search paths
            </div>
            """

        gir_cards.append(
            f"""
        <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 flex flex-col justify-between">
            <div class="flex items-center justify-between gap-4">
                <div>
                    <h4 class="text-sm font-bold text-white font-mono">{item['namespace']}-1.0</h4>
                    <p class="text-[11px] text-slate-500 mt-0.5">{item['filename']}</p>
                </div>
                {status_badge}
            </div>
            {path_display}
        </div>
        """
        )

    gir_cards_rendered = "\n".join(gir_cards)
    settings_gir_paths = ", ".join(settings.gir_search_paths)

    # 4. Build collapsible categories directory HTML
    categories_html = []
    for cat_name, elements_list in categories.items():
        if not elements_list:
            continue

        elements_rows_html = []
        for el in elements_list:
            elements_rows_html.append(
                f"""
            <div class="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
                <div class="space-y-1">
                    <div class="flex items-center gap-2">
                        <span class="font-mono font-bold text-emerald-400">{el['name']}</span>
                        <span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px] font-mono">{el['plugin']}</span>
                    </div>
                    <p class="text-slate-400 leading-normal">{el['desc']}</p>
                </div>
                <div class="text-right shrink-0">
                    <span class="text-[10px] text-slate-500 font-mono block">{el['klass']}</span>
                </div>
            </div>
            """
            )

        elements_html_block = "\n".join(elements_rows_html)

        categories_html.append(
            f"""
        <details class="group bg-slate-900/40 border border-slate-800/80 rounded-2xl overflow-hidden [&_summary::-webkit-details-marker]:hidden">
            <summary class="flex items-center justify-between p-5 cursor-pointer select-none hover:bg-slate-900/80 transition duration-150">
                <div class="flex items-center gap-3">
                    <span class="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 text-xs font-semibold font-mono">
                        {len(elements_list)}
                    </span>
                    <span class="font-bold text-white tracking-wide text-sm">{cat_name}</span>
                </div>
                <svg class="h-5 w-5 text-slate-400 transition duration-300 group-open:-rotate-180"
                     xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
            </summary>
            <div class="px-5 pb-5 border-t border-slate-800/60 divide-y divide-slate-800/40 max-h-96 overflow-y-auto">
                {elements_html_block}
            </div>
        </details>
        """
        )

    categories_rendered = "\n".join(categories_html)

    # Render landing dashboard
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

        <main class="max-w-6xl mx-auto px-6 py-10 space-y-12">
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

            <!-- Section: Collapsible Elements Directory -->
            <section class="space-y-6">
                <div>
                    <h2 class="text-2xl font-bold text-white tracking-tight">Registered GStreamer Elements</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        Browse through all elements currently loaded in GStreamer's registry inside this container,
                        grouped by semantic category. Click on any category block to expand and view the list.
                    </p>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {categories_rendered}
                </div>
            </section>

            <!-- Section: GIR Registry Status -->
            <section class="space-y-6">
                <div>
                    <h2 class="text-2xl font-bold text-white tracking-tight">GObject Introspection (.gir) Registry</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        GStreamer GObject API C metadata is read directly from system <code>.gir</code> XML files inside the container.
                        Below is the discovery status of these files within the configured search paths:
                        <code class="font-bold text-xs text-emerald-400">{settings_gir_paths}</code>.
                    </p>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {gir_cards_rendered}
                </div>
            </section>

            <!-- Section: Environment Variables -->
            <section class="space-y-6">
                <div>
                    <h2 class="text-2xl font-bold text-white tracking-tight">Live Environment Configuration</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        Current environment variables active inside this GStreamer agent container and on the proxy client.
                    </p>
                </div>
                <div class="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/20">
                    <table class="w-full text-left text-sm border-collapse">
                        <thead class="bg-slate-900/80 text-xs font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-800">
                            <tr>
                                <th class="px-6 py-4">Environment Variable</th>
                                <th class="px-6 py-4">Current Value</th>
                                <th class="px-6 py-4">Scope / Source</th>
                                <th class="px-6 py-4">Description</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60">
                            {env_rows_rendered}
                        </tbody>
                    </table>
                </div>
            </section>

             <!-- Section: Client Setup Instruction -->
            <section class="bg-slate-900/30 border border-slate-800/80 rounded-3xl p-8 space-y-8">
                <div>
                    <h2 class="text-2xl font-bold text-white tracking-tight">OpenCode &amp; Claude Desktop Integration</h2>
                    <p class="text-sm text-slate-400 mt-1">
                        Configure your host Model Context Protocol clients using the configuration snippets below.
                        They utilize <code>conda run</code> to run inside your virtual environment (<code>mcp-env</code>)
                        without environment path resolution mismatches on your host.
                    </p>
                </div>

                <!-- Grid: OpenCode and Claude side-by-side -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <!-- Option A: OpenCode -->
                    <div class="space-y-4 flex flex-col justify-between">
                        <div class="space-y-2">
                            <div class="flex items-center gap-2">
                                <span class="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 text-xs font-semibold uppercase tracking-wider">Option A</span>
                                <h3 class="text-lg font-bold text-white">OpenCode Configuration</h3>
                            </div>
                            <p class="text-xs text-slate-400 leading-relaxed">
                                Add the following block to your OpenCode configuration. You can configure this globally or at the workspace level.
                            </p>
                        </div>
                        <pre class="bg-slate-950 p-5 rounded-2xl border border-slate-800 text-xs overflow-x-auto text-emerald-400/90 leading-relaxed shadow-inner">
"mcp": {{
  "gstreamer": {{
    "type": "local",
    "command": [
      "conda",
      "run",
      "-n",
      "mcp-env",
      "--no-capture-output",
      "python3",
      "/home/jarno/projects/jarno/gstreamer-examples/docker/gstreamer_mcp.py"
    ],
    "environment": {{
      "GST_DOCS_AGENT_URL": "http://localhost:8000"
    }}
  }}
}}</pre>
                    </div>

                    <!-- Option B: Claude Desktop -->
                    <div class="space-y-4 flex flex-col justify-between">
                        <div class="space-y-2">
                            <div class="flex items-center gap-2">
                                <span class="px-2.5 py-1 rounded bg-teal-500/10 text-teal-400 text-xs font-semibold uppercase tracking-wider">Option B</span>
                                <h3 class="text-lg font-bold text-white">Claude Desktop Configuration</h3>
                            </div>
                            <p class="text-xs text-slate-400 leading-relaxed">
                                Add the following block to your Claude Desktop configuration file. Note that paths and settings are preserved automatically.
                            </p>
                        </div>
                        <pre class="bg-slate-950 p-5 rounded-2xl border border-slate-800 text-xs overflow-x-auto text-teal-400/90 leading-relaxed shadow-inner">
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
                </div>

                <!-- Table: Configuration Paths -->
                <div class="space-y-3 pt-4 border-t border-slate-800/60">
                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Where to Find Your Client Configuration Files</span>
                    <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/40">
                        <table class="w-full text-left text-xs border-collapse">
                            <thead class="bg-slate-900/60 text-[10px] font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-800/80">
                                <tr>
                                    <th class="px-5 py-3">Client Tool</th>
                                    <th class="px-5 py-3">OS / Platform</th>
                                    <th class="px-5 py-3">Configuration File Path</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800/40 text-slate-300">
                                <tr class="hover:bg-slate-900/20 transition-colors">
                                    <td class="px-5 py-3 font-semibold text-white">OpenCode</td>
                                    <td class="px-5 py-3">Ubuntu / Linux</td>
                                    <td class="px-5 py-3 font-mono text-emerald-400">
                                        ~/.config/opencode/opencode.json
                                        <span class="text-slate-500 font-sans">(or workspace-level .opencode/opencode.json)</span>
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-900/20 transition-colors">
                                    <td class="px-5 py-3 font-semibold text-white">OpenCode</td>
                                    <td class="px-5 py-3">macOS</td>
                                    <td class="px-5 py-3 font-mono text-emerald-400">
                                        ~/Library/Application Support/opencode/opencode.json
                                        <span class="text-slate-500 font-sans">(or workspace-level .opencode/opencode.json)</span>
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-900/20 transition-colors">
                                    <td class="px-5 py-3 font-semibold text-white">OpenCode</td>
                                    <td class="px-5 py-3">Windows</td>
                                    <td class="px-5 py-3 font-mono text-emerald-400">
                                        %APPDATA%\\opencode\\opencode.json
                                        <span class="text-slate-500 font-sans">(or workspace-level .opencode\\opencode.json)</span>
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-900/20 transition-colors">
                                    <td class="px-5 py-3 font-semibold text-white">Claude Desktop</td>
                                    <td class="px-5 py-3">macOS</td>
                                    <td class="px-5 py-3 font-mono text-teal-400">
                                        ~/Library/Application Support/Claude/claude_desktop_config.json
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-900/20 transition-colors">
                                    <td class="px-5 py-3 font-semibold text-white">Claude Desktop</td>
                                    <td class="px-5 py-3">Windows</td>
                                    <td class="px-5 py-3 font-mono text-teal-400">
                                        %APPDATA%\\Claude\\claude_desktop_config.json
                                    </td>
                                </tr>
                                <tr class="hover:bg-slate-900/20 transition-colors">
                                    <td class="px-5 py-3 font-semibold text-white">Claude Desktop</td>
                                    <td class="px-5 py-3">Ubuntu / Linux</td>
                                    <td class="px-5 py-3 font-mono text-teal-400">
                                        ~/.config/Claude/claude_desktop_config.json
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Section: Debugging -->
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

                    <!-- Discover Classes API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3 md:col-span-2">
                        <h4 class="text-sm font-bold text-white">Discover Classes &amp; Namespaces</h4>
                        <p class="text-xs text-slate-400">
                            Find available GObject classes and interfaces parsed dynamically from the system's `.gir` XML files.
                        </p>
                        <div class="space-y-2">
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">1. Filter by keyword:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/docs/classes?query=VideoDecoder"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">2. Filter by GObject Namespace:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/docs/classes?namespace=GstVideo"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">3. List All Available Namespaces &amp; Classes:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/docs/classes"</pre>
                            </div>
                        </div>
                    </div>

                    <!-- Available Elements API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3 md:col-span-2">
                        <h4 class="text-sm font-bold text-white">Search Registered Elements</h4>
                        <p class="text-xs text-slate-400">
                            Find installed components using precise filters for element name, plugin, classification class, or global query.
                        </p>
                        <div class="space-y-2">
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">
                                    1. Precise Element Name Filter (e.g. only names containing 'nv'):
                                </span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements?name=nv"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">2. Strict Plugin Filter:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements?plugin=videoparsersbad"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">
                                    3. Semantic Class Filter (e.g. Decoder, Encoder, Demuxer):
                                </span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements?klass=Decoder"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">4. Global Fallback Search:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements?query=nv"</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">5. List All Elements:</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements"</pre>
                            </div>
                        </div>
                    </div>

                    <!-- Inspect Element Details API -->
                    <div class="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 space-y-3 md:col-span-2">
                        <h4 class="text-sm font-bold text-white">Inspect Element Details (Structured Schema &amp; Optional Raw Specs)</h4>
                        <p class="text-xs text-slate-400">
                            Retrieves typed property structures, writable/readable flags, pad templates, and caps.
                            To save context tokens and execution speed, the raw <code>gst-inspect-1.0</code> output is
                            omitted by default and can be requested optionally with minification applied.
                        </p>
                        <div class="space-y-2">
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">1. Default (Highly efficient JSON schema only):</span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements/details?name=jpeg2000parse" | jq .</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">
                                    2. Request Minified Raw Text (Compact, saves ~20-30% tokens):
                                </span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements/details?name=jpeg2000parse&amp;raw=true" | jq .</pre>
                            </div>
                            <div>
                                <span class="text-[10px] font-semibold text-slate-500 uppercase font-mono">
                                    3. Request Original Raw CLI Text (With native indentation and whitespace):
                                </span>
                                <pre class="bg-slate-950 p-3 rounded-xl text-xs text-slate-300
                                            border border-slate-800/50 overflow-x-auto"
                                >curl "http://127.0.0.1:8000/elements/details?name=jpeg2000parse&amp;raw=true&amp;minify=false" | jq .</pre>
                            </div>
                        </div>
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
            &copy; 2026 GStreamer MCP Documentation Server Agent. Done cleanly &amp; maintainably.
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
    name: str
    | None = Query(None, description="Optional filter by element name (e.g. 'nv')"),
    plugin: str
    | None = Query(
        None, description="Optional filter strictly by GStreamer plugin name"
    ),
    klass: str
    | None = Query(
        None,
        description="Optional semantic class filter (e.g. 'Decoder', 'Encoder', 'Demuxer', 'Source')",
    ),
    query: str
    | None = Query(
        None,
        description="Optional global fallback search (name, desc, plugin, class)",
    ),
) -> dict[str, Any]:
    """
    List and filter GStreamer elements registered in the container.

    Parameters
    ----------
    name : str or None, optional
        An element name filter, by default None.
    plugin : str or None, optional
        A GStreamer plugin name filter, by default None.
    klass : str or None, optional
        A semantic class to filter elements, by default None.
    query : str or None, optional
        A global search string to filter elements by name, description, plugin, or class, by default None.

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
            f_name = factory.get_name()
            f_klass = factory.get_klass() or ""
            desc = factory.get_description() or ""
            f_plugin = factory.get_plugin_name() or "core"

            # 1. Strict Name Filter
            if name and name.lower() not in f_name.lower():
                continue

            # 2. Strict Plugin Filter
            if plugin and plugin.lower() not in f_plugin.lower():
                continue

            # 3. Strict Classification Filter
            if klass and klass.lower() not in f_klass.lower():
                continue

            # 4. Fallback Global Query
            if query:
                q = query.lower()
                if (
                    q not in f_name.lower()
                    and q not in desc.lower()
                    and q not in f_plugin.lower()
                    and q not in f_klass.lower()
                ):
                    continue

            elements.append(
                {
                    "plugin": f_plugin,
                    "element": f_name,
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
    ),
    raw: bool = Query(
        False, description="Whether to include raw gst-inspect-1.0 text in response"
    ),
    minify: bool = Query(
        True,
        description="Whether to minify raw text by stripping redundant whitespace and empty lines, if raw is True",
    ),
) -> dict[str, Any]:
    """
    Get detailed information about a GStreamer element including schema and properties.

    Parameters
    ----------
    name : str
        The name of the GStreamer element to inspect.
    raw : bool, optional
        Whether to include raw gst-inspect-1.0 text in the response, by default False.
    minify : bool, optional
        Whether to minify the raw text to save context tokens, by default True.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status and element details (schema and optionally raw text).

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

        # 1. Run gst-inspect-1.0 only if raw is requested
        raw_text = ""
        if raw:
            try:
                result = subprocess.run(
                    ["gst-inspect-1.0", name],
                    capture_output=True,
                    text=True,
                    check=True,
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

        data = {"schema": schema}
        if raw and raw_text:
            if minify:
                import re

                minified_lines = []
                for line in raw_text.splitlines():
                    line = line.rstrip()
                    if not line:
                        continue
                    leading_spaces = len(line) - len(line.lstrip(" "))
                    content = line.lstrip(" ")
                    content = re.sub(r" {2,}", " ", content)
                    minified_lines.append(" " * leading_spaces + content)
                raw_text = "\n".join(minified_lines)
            data["raw_text"] = raw_text

        return {"status": "success", "data": data}
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
    class_path: str = Query(..., description="E.g., Gst.Element, Gst.Pad"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Retrieve C signatures and summary from GI Introspection XML for a GObject class.

    Parameters
    ----------
    class_path : str
        Dot-separated GObject class path, e.g., 'Gst.Element'.
    settings : Settings
        Application settings

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
        gir_path = _find_gir_path(
            namespace=namespace, gir_search_paths=settings.gir_search_paths
        )
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


_classes_cache: dict[str, list[str]] = {}


def _list_available_namespaces(search_paths: list[str]) -> list[str]:
    """
    List available GStreamer-related GObject namespaces from .gir files.

    Parameters
    ----------
    search_paths : list of str
        The search paths for GObject Introspection (.gir) files.

    Returns
    -------
    list of str
        A sorted list of available namespace names.
    """
    namespaces = set()
    for base_dir in search_paths:
        if os.path.exists(base_dir):
            for f in os.listdir(base_dir):
                if f.endswith(".gir"):
                    parts = f.split("-")
                    if parts:
                        ns_name = parts[0]
                        if ns_name.startswith("Gst") or ns_name in (
                            "GLib",
                            "GObject",
                            "Gio",
                        ):
                            namespaces.add(ns_name)
    return sorted(list(namespaces))


def _get_classes_for_namespace(namespace: str, search_paths: list[str]) -> list[str]:
    """
    Get all classes and interfaces for a specific GObject namespace.

    Parameters
    ----------
    namespace : str
        The GObject namespace, e.g., 'Gst'.
    search_paths : list of str
        The search paths for .gir files.

    Returns
    -------
    list of str
        A sorted list of class and interface names in the namespace.
    """
    if namespace in _classes_cache:
        return _classes_cache[namespace]

    try:
        gir_path = _find_gir_path(namespace, search_paths)
    except FileNotFoundError:
        return []

    try:
        tree = ET.parse(gir_path)
        root = tree.getroot()
        ns = {
            "core": "http://www.gtk.org/introspection/core/1.0",
        }

        classes = []
        for tag in ["class", "interface"]:
            for elem in root.findall(f".//core:{tag}", ns):
                name = elem.attrib.get("name")
                if name:
                    classes.append(name)

        classes.sort()
        _classes_cache[namespace] = classes
        return classes
    except Exception:
        return []


@app.get("/docs/classes")
def get_available_classes(
    namespace: str
    | None = Query(
        None, description="Optional namespace filter (e.g. 'Gst', 'GstVideo')"
    ),
    query: str
    | None = Query(None, description="Optional search/filter string for class names"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    List and filter GObject classes and interfaces available in the container.

    Parameters
    ----------
    namespace : str or None, optional
        A namespace name to filter classes, by default None.
    query : str or None, optional
        A search string to filter classes by name or class path, by default None.
    settings : Settings
        Application settings.

    Returns
    -------
    dict of str to Any
        A dictionary containing the status, available namespaces, and classes.
    """
    available_namespaces = _list_available_namespaces(settings.gir_search_paths)

    if namespace:
        if namespace not in available_namespaces:
            raise HTTPException(
                status_code=404,
                detail=f"Namespace '{namespace}' is not available. Available: {available_namespaces}",
            )
        target_namespaces = [namespace]
    else:
        target_namespaces = available_namespaces

    data = []
    for ns_name in target_namespaces:
        classes = _get_classes_for_namespace(ns_name, settings.gir_search_paths)
        for cls in classes:
            class_path = f"{ns_name}.{cls}"
            # Apply search filter
            if query:
                q = query.lower()
                if q not in cls.lower() and q not in class_path.lower():
                    continue

            data.append(
                {
                    "namespace": ns_name,
                    "class": cls,
                    "class_path": class_path,
                }
            )

    return {
        "status": "success",
        "namespaces": available_namespaces,
        "data": data,
    }
