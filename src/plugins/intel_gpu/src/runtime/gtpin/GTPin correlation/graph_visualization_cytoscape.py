#!/usr/bin/env python3
"""
Generate Cytoscape.js DAG views from graph-prep execution-unit bundles.

The current primary view is a layered DAG over the schema:
component_group -> op_type -> layer -> primitive -> kernel

Dependency edges between primitives are preserved as an optional overlay.
They add correlation context, but they are excluded from the primary layout
pass so the hierarchy stays readable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


NODE_ORDER = {
    "component_group": 0,
    "op_type": 1,
    "origin_op": 2,
    "primitive": 3,
    "kernel": 4,
}


DEFAULT_CYTOSCAPE_URL = "https://unpkg.com/cytoscape@3.34.1/dist/cytoscape.min.js"
DEFAULT_CY_DAGRE_URL = "https://unpkg.com/cytoscape-dagre@4.0.1/cytoscape-dagre.js"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: str, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def detect_bundle_prefix(unit_dir: Path) -> str:
    summaries = sorted(unit_dir.glob("*_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary bundle found in {unit_dir}")
    return summaries[0].name[: -len("_summary.json")]


def available_unit_dirs(graph_prep_root: Path) -> List[Path]:
    return sorted(path for path in graph_prep_root.iterdir() if path.is_dir() and path.name.startswith("n"))


def build_parent_lookup(nodes: List[Dict[str, str]]) -> Dict[str, str]:
    return {str(node["node_id"]): str(node.get("parent_node_id", "")) for node in nodes}


def collect_ancestors(node_id: str, parent_by_id: Dict[str, str]) -> List[str]:
    ancestors: List[str] = []
    current = parent_by_id.get(node_id, "")
    while current:
        ancestors.append(current)
        current = parent_by_id.get(current, "")
    return ancestors


def build_children_lookup(nodes: List[Dict[str, str]]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for node in nodes:
        parent_id = str(node.get("parent_node_id", ""))
        if parent_id:
            children[parent_id].append(str(node["node_id"]))
    return children


def collect_descendants(root_id: str, children_by_id: Dict[str, List[str]]) -> List[str]:
    descendants: List[str] = []
    stack = list(children_by_id.get(root_id, []))
    while stack:
        current = stack.pop()
        descendants.append(current)
        stack.extend(children_by_id.get(current, []))
    return descendants


def node_weight(total_cycles: int, floor: float, ceiling: float) -> float:
    if total_cycles <= 0:
        return floor
    scaled = math.log10(total_cycles + 1)
    return clamp(floor + scaled * 6.0, floor, ceiling)


def dependency_cycle_count(nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> int:
    primitive_ids = {
        str(node["node_id"])
        for node in nodes
        if str(node.get("node_type", "")) == "primitive"
    }
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if str(edge.get("edge_type", "")) != "primitive_dependency":
            continue
        source = str(edge.get("source_node_id", ""))
        target = str(edge.get("target_node_id", ""))
        if source in primitive_ids and target in primitive_ids:
            adjacency[source].append(target)

    seen: Set[str] = set()
    active: Set[str] = set()
    cycle_hits = 0

    def visit(node_id: str) -> None:
        nonlocal cycle_hits
        seen.add(node_id)
        active.add(node_id)
        for child in adjacency.get(node_id, []):
            if child not in seen:
                visit(child)
            elif child in active:
                cycle_hits += 1
        active.remove(node_id)

    for node_id in primitive_ids:
        if node_id not in seen:
            visit(node_id)

    return cycle_hits


def build_elements(nodes: List[Dict[str, str]], edges: List[Dict[str, str]], summary: Dict[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    parent_by_id = build_parent_lookup(nodes)
    children_by_id = build_children_lookup(nodes)
    node_by_id = {str(node["node_id"]): node for node in nodes}
    max_cycles = max((to_int(node.get("total_cycles", "0")) for node in nodes), default=0)

    elements: List[Dict[str, object]] = []
    type_counts: Dict[str, int] = defaultdict(int)

    for node in nodes:
        node_id = str(node["node_id"])
        node_type = str(node.get("node_type", ""))
        total_cycles = to_int(node.get("total_cycles", "0"))
        dispatch_rows = to_int(node.get("dispatch_rows", "0"))
        kernel_count = to_int(node.get("kernel_entry_count", "0"))
        type_counts[node_type] += 1

        ancestors = collect_ancestors(node_id, parent_by_id)
        descendants = collect_descendants(node_id, children_by_id)
        elements.append({
            "data": {
                "id": node_id,
                "label": str(node.get("label", "")),
                "node_type": node_type,
                "level": NODE_ORDER.get(node_type, 99),
                "execution_unit_key": str(node.get("execution_unit_key", "")),
                "component_path": str(node.get("component_path", "")),
                "origin_op_name": str(node.get("origin_op_name", "")),
                "origin_op_type_name": str(node.get("origin_op_type_name", "")),
                "resolved_op_type_name": str(node.get("resolved_op_type_name", "")),
                "primitive_id": str(node.get("primitive_id", "")),
                "primitive_type": str(node.get("primitive_type", "")),
                "implementation": str(node.get("implementation", "")),
                "mapping_status": str(node.get("mapping_status", "")),
                "topology_mode": str(node.get("topology_mode", "")),
                "dispatch_rows": dispatch_rows,
                "kernel_entry_count": kernel_count,
                "kernel_entries": str(node.get("kernel_entries", "")),
                "total_cycles": total_cycles,
                "layer_children": to_int(node.get("layer_children", "0")),
                "primitive_children": to_int(node.get("primitive_children", "0")),
                "dependencies": str(node.get("dependencies", "")),
                "users": str(node.get("users", "")),
                "fused_ids": str(node.get("fused_ids", "")),
                "weight": node_weight(total_cycles, 26.0 if node_type == "component_group" else 18.0, 84.0 if node_type == "kernel" else 54.0),
                "heat": 0.0 if max_cycles <= 0 else round(total_cycles / max_cycles, 6),
                "ancestor_count": len(ancestors),
                "descendant_count": len(descendants),
                "detail_lines": [
                    f"type: {node_type}",
                    f"label: {str(node.get('label', ''))}",
                    f"cycles: {total_cycles}",
                    f"dispatch_rows: {dispatch_rows}",
                    f"mapping_status: {str(node.get('mapping_status', ''))}",
                    f"resolved_op_type_name: {str(node.get('resolved_op_type_name', ''))}",
                    f"component_path: {str(node.get('component_path', ''))}",
                    f"primitive_id: {str(node.get('primitive_id', ''))}",
                    f"implementation: {str(node.get('implementation', ''))}",
                    f"kernel_entries: {str(node.get('kernel_entries', ''))}",
                ],
            },
            "classes": f"node-{node_type}",
        })

    dependency_count = 0
    hierarchy_count = 0
    for edge in edges:
        edge_type = str(edge.get("edge_type", ""))
        semantic = str(edge.get("semantic", ""))
        source_id = str(edge.get("source_node_id", ""))
        target_id = str(edge.get("target_node_id", ""))
        if source_id not in node_by_id or target_id not in node_by_id:
            continue

        if semantic == "dependency":
            dependency_count += 1
        else:
            hierarchy_count += 1

        elements.append({
            "data": {
                "id": f"{source_id}--{target_id}--{edge_type}",
                "source": source_id,
                "target": target_id,
                "edge_type": edge_type,
                "semantic": semantic,
                "source_label": str(edge.get("source_label", "")),
                "target_label": str(edge.get("target_label", "")),
            },
            "classes": f"edge-{semantic} edge-{edge_type}",
        })

    graph_meta = {
        "execution_unit_key": summary.get("execution_unit_key", ""),
        "net_id": summary.get("net_id", ""),
        "iteration": summary.get("iteration", ""),
        "all_nodes": summary.get("all_nodes", len(nodes)),
        "all_edges": summary.get("all_edges", len(edges)),
        "filtered_nodes": summary.get("filtered_nodes", len(nodes)),
        "filtered_edges": summary.get("filtered_edges", len(edges)),
        "topdown_mode": summary.get("topdown_mode", ""),
        "type_counts": dict(sorted(type_counts.items())),
        "hierarchy_edge_count": hierarchy_count,
        "dependency_edge_count": dependency_count,
        "dependency_cycle_hits": dependency_cycle_count(nodes, edges),
    }

    return elements, graph_meta


def render_html(
    title: str,
    elements: List[Dict[str, object]],
    graph_meta: Dict[str, object],
    use_dagre: bool,
    cytoscape_url: str,
    cy_dagre_url: str,
) -> str:
    elements_json = json.dumps(elements, ensure_ascii=False)
    meta_json = json.dumps(graph_meta, ensure_ascii=False)
    layout_name = "dagre" if use_dagre else "breadthfirst"
    dagre_script_tag = f'<script src="{cy_dagre_url}"></script>' if use_dagre else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="{cytoscape_url}"></script>
  {dagre_script_tag}
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: rgba(255, 252, 246, 0.96);
      --border: #c9bea7;
      --ink: #1d1d1b;
      --muted: #5f5a52;
      --accent: #26547c;
      --accent-soft: #d7e5ef;
      --warn: #b76e2b;
      --shadow: 0 22px 48px rgba(69, 54, 22, 0.12);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(219, 187, 130, 0.24), transparent 28%),
        radial-gradient(circle at bottom right, rgba(38, 84, 124, 0.14), transparent 30%),
        var(--bg);
    }}
    .shell {{
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      padding: 22px;
      background: var(--panel);
      border-right: 1px solid var(--border);
      box-shadow: var(--shadow);
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      z-index: 3;
    }}
    .main {{
      min-width: 0;
      padding: 18px;
    }}
    .card {{
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.86);
      border-radius: 16px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.15;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .lede {{
      margin: 0;
      line-height: 1.45;
      color: var(--muted);
      font-size: 14px;
    }}
    .stat-list {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }}
    .stat {{
      border: 1px solid #ddd4c1;
      border-radius: 12px;
      padding: 10px;
      background: #fffdfa;
    }}
    .stat .k {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .v {{
      display: block;
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
    }}
    .control {{
      display: block;
      margin: 10px 0;
      font-size: 14px;
    }}
    .control input[type="checkbox"] {{
      margin-right: 8px;
    }}
    .search {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      font-size: 14px;
      background: #fffdfa;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      font-size: 14px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(0, 0, 0, 0.18);
      flex: none;
    }}
    .detail {{
      white-space: pre-wrap;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #2a2a28;
      margin: 0;
    }}
    .note {{
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
      margin: 0;
    }}
    #cy {{
      width: 100%;
      height: calc(100vh - 36px);
      min-height: 860px;
      border: 1px solid var(--border);
      border-radius: 24px;
      background:
        linear-gradient(90deg, rgba(116, 98, 64, 0.06) 1px, transparent 1px) 0 0 / 240px 240px,
        linear-gradient(180deg, rgba(116, 98, 64, 0.05) 1px, transparent 1px) 0 0 / 240px 240px,
        linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(249, 244, 235, 0.98));
      box-shadow: var(--shadow);
    }}
    .footer {{
      margin-top: 14px;
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 1200px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: relative;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--border);
      }}
      #cy {{
        height: 78vh;
        min-height: 680px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="card">
        <h1>{title}</h1>
        <p class="lede">Layered Cytoscape DAG view over the current correlation schema. Hierarchy drives placement; primitive dependency edges are an overlay for higher-level context.</p>
      </div>

      <div class="card">
        <h2>Summary</h2>
        <div class="stat-list">
          <div class="stat"><span class="k">Execution Unit</span><span class="v" id="stat-unit"></span></div>
          <div class="stat"><span class="k">Topology Mode</span><span class="v" id="stat-topology"></span></div>
          <div class="stat"><span class="k">Nodes</span><span class="v" id="stat-nodes"></span></div>
          <div class="stat"><span class="k">Edges</span><span class="v" id="stat-edges"></span></div>
          <div class="stat"><span class="k">Hierarchy Edges</span><span class="v" id="stat-hierarchy"></span></div>
          <div class="stat"><span class="k">Dependency Edges</span><span class="v" id="stat-deps"></span></div>
        </div>
      </div>

      <div class="card">
        <h2>Controls</h2>
        <input id="search" class="search" type="text" placeholder="Search label, primitive, op type">
        <label class="control"><input id="toggleDeps" type="checkbox" checked>Show primitive dependency overlay</label>
        <label class="control"><input id="toggleKernelLabels" type="checkbox">Show kernel labels</label>
        <label class="control"><input id="togglePrimitiveLabels" type="checkbox" checked>Show primitive labels</label>
        <label class="control"><input id="fitGraph" type="checkbox" checked>Refit graph after layout updates</label>
        <button id="resetRootView" type="button" style="margin-top:10px;width:100%;padding:10px 12px;border:1px solid var(--border);border-radius:12px;background:#fffdfa;color:var(--ink);font:600 14px 'Segoe UI',sans-serif;cursor:pointer;">Reset To Root View</button>
      </div>

      <div class="card">
        <h2>Legend</h2>
        <div class="legend-row"><span class="swatch" style="background:#517f72"></span>component_group</div>
        <div class="legend-row"><span class="swatch" style="background:#3f6ea3"></span>op_type</div>
        <div class="legend-row"><span class="swatch" style="background:#d59a3a"></span>layer</div>
        <div class="legend-row"><span class="swatch" style="background:#4e9b63"></span>primitive</div>
        <div class="legend-row"><span class="swatch" style="background:#c45f3c"></span>kernel</div>
        <div class="legend-row"><span class="swatch" style="background:#b76e2b"></span>primitive dependency edge</div>
      </div>

      <div class="card">
        <h2>Selection</h2>
        <pre id="detail" class="detail">Select a node or edge to inspect details.</pre>
      </div>

      <div class="card">
        <h2>Graph Notes</h2>
        <p class="note" id="graph-note"></p>
      </div>

      <div class="footer" id="layout-footer">Requested layout: {layout_name}. Direction: left to right.</div>
    </aside>
    <main class="main">
      <div id="cy"></div>
    </main>
  </div>

  <script>
    const elements = {elements_json};
    const graphMeta = {meta_json};
    const requestedLayoutName = "{layout_name}";
    const detailEl = document.getElementById("detail");
    const searchEl = document.getElementById("search");
    const depsToggle = document.getElementById("toggleDeps");
    const kernelLabelToggle = document.getElementById("toggleKernelLabels");
    const primitiveLabelToggle = document.getElementById("togglePrimitiveLabels");
    const fitToggle = document.getElementById("fitGraph");
    const resetRootViewButton = document.getElementById("resetRootView");
    const graphNoteEl = document.getElementById("graph-note");
    const layoutFooterEl = document.getElementById("layout-footer");
    const cyContainer = document.getElementById("cy");

    function fillSummary() {{
      document.getElementById("stat-unit").textContent = graphMeta.execution_unit_key || "-";
      document.getElementById("stat-topology").textContent = graphMeta.topdown_mode || "-";
      document.getElementById("stat-nodes").textContent = String(graphMeta.filtered_nodes || graphMeta.all_nodes || 0);
      document.getElementById("stat-edges").textContent = String(graphMeta.filtered_edges || graphMeta.all_edges || 0);
      document.getElementById("stat-hierarchy").textContent = String(graphMeta.hierarchy_edge_count || 0);
      document.getElementById("stat-deps").textContent = String(graphMeta.dependency_edge_count || 0);

      const cycleHits = Number(graphMeta.dependency_cycle_hits || 0);
      const note = cycleHits > 0
        ? "Dependency edges introduce at least " + cycleHits + " back-edge signal(s), so the full graph is no longer a strict DAG unless the dependency overlay is hidden."
        : "The dependency overlay stays acyclic on this execution unit, so the whole view remains DAG-consistent.";
      graphNoteEl.textContent = note;
    }}

    function showGraphError(message) {{
      cyContainer.innerHTML = `
        <div style="display:flex;height:100%;align-items:center;justify-content:center;padding:32px;">
          <div style="max-width:760px;background:rgba(255,253,247,0.96);border:1px solid #c9bea7;border-radius:18px;padding:22px;box-shadow:0 18px 40px rgba(72,56,26,0.1);font:14px/1.55 'Segoe UI',sans-serif;color:#1d1d1b;">
            <div style="font-size:18px;font-weight:700;margin-bottom:10px;">Graph initialization failed</div>
            <div>${{message}}</div>
          </div>
        </div>
      `;
      detailEl.textContent = message;
    }}

    fillSummary();

    const hasCytoscape = typeof window.cytoscape !== "undefined";
    let activeLayoutName = requestedLayoutName;

    if (!hasCytoscape) {{
      layoutFooterEl.textContent = "Cytoscape.js failed to load.";
      graphNoteEl.textContent = "The page data is present, but the browser could not load Cytoscape.js. This usually means the CDN scripts were blocked or unavailable while opening the local HTML file.";
      showGraphError("Cytoscape.js did not load. The HTML shell rendered, but the graph library was unavailable, so no nodes or zoom interaction could initialize.");
    }} else {{
      let cy;
      try {{
        cy = cytoscape({{
          container: cyContainer,
          elements,
          layout: {{
            name: "preset",
            fit: false,
            padding: 0,
            animate: false
          }},
          wheelSensitivity: 0.18,
          userZoomingEnabled: true,
          userPanningEnabled: true,
          boxSelectionEnabled: false,
          minZoom: 0.02,
          maxZoom: 8,
          style: [
        {{
          selector: "node",
          style: {{
            "label": "data(label)",
            "font-size": 11,
            "font-weight": 600,
            "text-wrap": "wrap",
            "text-max-width": 180,
            "text-halign": "center",
            "text-valign": "center",
            "color": "#1e1e1b",
            "background-color": "#8ca38a",
            "border-width": 1.5,
            "border-color": "#35504a",
            "width": "data(weight)",
            "height": "data(weight)",
            "padding": "10px",
            "overlay-opacity": 0,
            "text-outline-width": 3,
            "text-outline-color": "rgba(255,255,255,0.78)"
          }}
        }},
        {{
          selector: ".node-component_group",
          style: {{
            "shape": "round-rectangle",
            "background-color": "#517f72",
            "border-color": "#27433b",
            "width": 120,
            "height": 48,
            "font-size": 13
          }}
        }},
        {{
          selector: ".node-op_type",
          style: {{
            "shape": "round-rectangle",
            "background-color": "#3f6ea3",
            "border-color": "#23415f",
            "width": 134,
            "height": 52,
            "font-size": 12,
            "color": "#f9fbff",
            "text-outline-color": "rgba(35,65,95,0.8)"
          }}
        }},
        {{
          selector: ".node-origin_op",
          style: {{
            "shape": "round-rectangle",
            "background-color": "#d59a3a",
            "border-color": "#835719",
            "width": 170,
            "height": 54,
            "font-size": 10
          }}
        }},
        {{
          selector: ".node-primitive",
          style: {{
            "shape": "ellipse",
            "background-color": "#4e9b63",
            "border-color": "#275333"
          }}
        }},
        {{
          selector: ".node-kernel",
          style: {{
            "shape": "diamond",
            "background-color": "#c45f3c",
            "border-color": "#722f18",
            "color": "#fff8f1",
            "text-outline-color": "rgba(114,47,24,0.86)",
            "font-size": 9
          }}
        }},
        {{
          selector: "edge",
          style: {{
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "line-color": "rgba(77, 76, 72, 0.34)",
            "target-arrow-color": "rgba(77, 76, 72, 0.34)",
            "width": 1.5,
            "overlay-opacity": 0
          }}
        }},
        {{
          selector: ".edge-hierarchy",
          style: {{
            "line-color": "rgba(70, 72, 78, 0.32)",
            "target-arrow-color": "rgba(70, 72, 78, 0.32)",
            "width": 1.5
          }}
        }},
        {{
          selector: ".edge-dependency",
          style: {{
            "line-color": "rgba(183, 110, 43, 0.74)",
            "target-arrow-color": "rgba(183, 110, 43, 0.74)",
            "line-style": "dashed",
            "width": 2.2,
            "curve-style": "unbundled-bezier",
            "control-point-distances": [-40, 40],
            "control-point-weights": [0.25, 0.75]
          }}
        }},
        {{
          selector: ".is-faded",
          style: {{
            "opacity": 0.12,
            "text-opacity": 0.1
          }}
        }},
        {{
          selector: ".is-match",
          style: {{
            "border-width": 4,
            "border-color": "#111111",
            "line-color": "#111111",
            "target-arrow-color": "#111111",
            "opacity": 1
          }}
        }},
        {{
          selector: ".hide-label",
          style: {{
            "label": ""
          }}
        }}
          ]
        }});
      }} catch (error) {{
        layoutFooterEl.textContent = "Graph creation failed.";
        graphNoteEl.textContent += " Initialization aborted before rendering, so interaction and zoom never became active.";
        showGraphError("Cytoscape threw an initialization error: " + String(error));
      }}

      if (cy) {{
        let dagreLooksAvailable = requestedLayoutName !== "dagre";
        if (requestedLayoutName === "dagre") {{
          try {{
            cy.layout({{ name: "dagre", animate: false }});
            dagreLooksAvailable = true;
          }} catch (error) {{
            dagreLooksAvailable = false;
          }}
        }}

        if (requestedLayoutName === "dagre" && !dagreLooksAvailable) {{
          activeLayoutName = "breadthfirst";
        }}

        const layoutOptions = activeLayoutName === "dagre"
          ? {{
              name: "dagre",
              rankDir: "LR",
              fit: true,
              padding: 80,
              nodeSep: 34,
              edgeSep: 14,
              rankSep: 170,
              animate: false,
              spacingFactor: 1.08,
              directed: true,
              avoidOverlap: true
            }}
          : {{
              name: "breadthfirst",
              directed: true,
              fit: true,
              padding: 80,
              spacingFactor: 1.08,
              avoidOverlap: true
            }};

        layoutFooterEl.textContent = "Requested layout: " + requestedLayoutName + ". Active layout: " + activeLayoutName + ". Direction: left to right.";
        if (requestedLayoutName === "dagre" && activeLayoutName !== "dagre") {{
          graphNoteEl.textContent += " Dagre layout was not available at runtime, so the page fell back to Cytoscape's built-in breadthfirst layout.";
        }}

        const activeLayout = cy.layout(layoutOptions);

        function rootNodes() {{
          return cy.nodes().filter((node) => node.data("node_type") === "component_group");
        }}

        function collectRootViewport(depth) {{
          let frontier = rootNodes();
          let collected = frontier;
          for (let index = 0; index < depth; index += 1) {{
            const next = frontier.outgoers(".edge-hierarchy").targets();
            if (next.empty()) {{
              break;
            }}
            collected = collected.union(next);
            frontier = next;
          }}
          return collected;
        }}

        function focusRootView() {{
          const roots = rootNodes();
          if (roots.empty()) {{
            cy.fit(undefined, 70);
            return;
          }}

          const viewportNodes = collectRootViewport(2);
          const bounds = viewportNodes.boundingBox();
          const containerWidth = cy.width();
          const containerHeight = cy.height();
          const boxWidth = Math.max(bounds.w, 1);
          const boxHeight = Math.max(bounds.h, 1);
          const fitZoomX = (containerWidth - 180) / boxWidth;
          const fitZoomY = (containerHeight - 140) / boxHeight;
          const nextZoom = Math.max(cy.minZoom(), Math.min(cy.maxZoom(), Math.min(fitZoomX, fitZoomY, 1.35)));
          const panX = 90 - (bounds.x1 * nextZoom);
          const panY = (containerHeight * 0.5) - (((bounds.y1 + bounds.y2) / 2) * nextZoom);

          cy.zoom(nextZoom);
          cy.pan({{ x: panX, y: panY }});
        }}

        function layoutLooksCollapsed() {{
          const bounds = cy.elements().boundingBox();
          const visibleNodeCount = cy.nodes(":visible").length;
          return visibleNodeCount > 100 && bounds.w < 420 && bounds.h < 220;
        }}

        function updateLabelDensity() {{
          cy.nodes(".node-kernel").toggleClass("hide-label", !kernelLabelToggle.checked);
          cy.nodes(".node-primitive").toggleClass("hide-label", !primitiveLabelToggle.checked);
        }}

        function updateDependencyVisibility() {{
          const visible = depsToggle.checked;
          cy.edges(".edge-dependency").style("display", visible ? "element" : "none");
          if (fitToggle.checked) {{
            focusRootView();
          }}
        }}

        function clearHighlight() {{
          cy.elements().removeClass("is-faded is-match");
        }}

        function applySearch() {{
          const query = searchEl.value.trim().toLowerCase();
          clearHighlight();
          if (!query) {{
            detailEl.textContent = "Select a node or edge to inspect details.";
            return;
          }}

          const matched = cy.elements().filter((ele) => {{
            if (ele.isNode()) {{
              const fields = [
                ele.data("label"),
                ele.data("primitive_id"),
                ele.data("origin_op_name"),
                ele.data("resolved_op_type_name"),
                ele.data("component_path")
              ];
              return fields.some((value) => String(value || "").toLowerCase().includes(query));
            }}
            const fields = [
              ele.data("edge_type"),
              ele.data("source_label"),
              ele.data("target_label")
            ];
            return fields.some((value) => String(value || "").toLowerCase().includes(query));
          }});

          cy.elements().addClass("is-faded");
          matched.removeClass("is-faded").addClass("is-match");
          matched.connectedEdges().removeClass("is-faded");
          matched.connectedNodes().removeClass("is-faded");

          if (matched.length > 0 && fitToggle.checked) {{
            cy.fit(matched, 100);
          }}

          detailEl.textContent = matched.length + " element(s) matched search: " + query;
        }}

        cy.on("tap", "node", (event) => {{
          const data = event.target.data();
          detailEl.textContent = data.detail_lines.join("\\n");
        }});

        cy.on("tap", "edge", (event) => {{
          const data = event.target.data();
          detailEl.textContent = [
            "edge_type: " + data.edge_type,
            "semantic: " + data.semantic,
            "source: " + data.source_label,
            "target: " + data.target_label
          ].join("\\n");
        }});

        cy.on("tap", (event) => {{
          if (event.target === cy) {{
            clearHighlight();
          }}
        }});

        let appliedInitialViewport = false;
        cy.on("layoutstop", () => {{
          if (!appliedInitialViewport) {{
            focusRootView();
            appliedInitialViewport = true;
          }} else if (fitToggle.checked) {{
            focusRootView();
          }}

          if (layoutLooksCollapsed()) {{
            graphNoteEl.textContent += " Layout warning: the resulting node bounding box is unusually small for this node count, so the active layout may have collapsed or failed to spread the graph.";
          }}
        }});

        searchEl.addEventListener("input", applySearch);
        depsToggle.addEventListener("change", updateDependencyVisibility);
        kernelLabelToggle.addEventListener("change", updateLabelDensity);
        primitiveLabelToggle.addEventListener("change", updateLabelDensity);
        resetRootViewButton.addEventListener("click", focusRootView);

        updateLabelDensity();
        updateDependencyVisibility();
        activeLayout.run();
      }}
    }}
  </script>
</body>
</html>
"""


def render_index(title: str, entries: List[Dict[str, str]]) -> str:
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td><a href=\"{entry['href']}\">{entry['execution_unit_key']}</a></td>"
            f"<td>{entry['iteration']}</td>"
            f"<td>{entry['nodes']}</td>"
            f"<td>{entry['edges']}</td>"
            f"<td>{entry['topdown_mode']}</td>"
            f"<td>{entry['notes']}</td>"
            "</tr>"
        )
    row_html = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      color: #1e1d1a;
      background:
        radial-gradient(circle at top left, rgba(213, 154, 58, 0.18), transparent 26%),
        linear-gradient(180deg, #f6f1e8 0%, #f4efe6 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    p {{
      margin: 0 0 22px;
      color: #5f5a52;
      line-height: 1.55;
      max-width: 860px;
    }}
    .card {{
      background: rgba(255, 253, 247, 0.94);
      border: 1px solid #c9bea7;
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(72, 56, 26, 0.1);
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid #e2d9c7;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #efe6d6;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
      color: #5f5a52;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    a {{
      color: #23415f;
      font-weight: 700;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>{title}</h1>
    <p>This index lists the generated Cytoscape DAG views. Each page uses the same layered schema and keeps primitive dependency edges as a toggleable overlay, so we can evaluate structural readability without committing to a heavier rendering stack.</p>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Execution Unit</th>
            <th>Iteration</th>
            <th>Nodes</th>
            <th>Edges</th>
            <th>Topology Mode</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {row_html}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def load_bundle(unit_dir: Path, bundle_prefix: str, mode: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    nodes_suffix = "nodes_filtered.csv" if mode == "filtered" else "nodes_all.csv"
    edges_suffix = "edges_filtered.csv" if mode == "filtered" else "edges_all.csv"
    nodes = read_csv(unit_dir / f"{bundle_prefix}_{nodes_suffix}")
    edges = read_csv(unit_dir / f"{bundle_prefix}_{edges_suffix}")
    summary = read_json(unit_dir / f"{bundle_prefix}_summary.json")
    return nodes, edges, summary


def write_outputs(
    graph_prep_root: Path,
    output_dir: Path,
    mode: str,
    use_dagre: bool,
    cytoscape_url: str,
    cy_dagre_url: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_entries: List[Dict[str, str]] = []

    for unit_dir in available_unit_dirs(graph_prep_root):
        bundle_prefix = detect_bundle_prefix(unit_dir)
        nodes, edges, summary = load_bundle(unit_dir, bundle_prefix, mode)
        elements, graph_meta = build_elements(nodes, edges, summary)

        page_name = f"{unit_dir.name}_{mode}_cytoscape_dag.html"
        page_path = output_dir / page_name
        json_path = output_dir / f"{unit_dir.name}_{mode}_cytoscape_dag.json"
        page_title = f"{unit_dir.name} DAG"

        page_path.write_text(
            render_html(
                title=page_title,
                elements=elements,
                graph_meta=graph_meta,
                use_dagre=use_dagre,
                cytoscape_url=cytoscape_url,
                cy_dagre_url=cy_dagre_url,
            ),
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps({"elements": elements, "graph_meta": graph_meta}, indent=2),
            encoding="utf-8",
        )

        notes = "overlay has dependency back-edges" if int(graph_meta["dependency_cycle_hits"]) > 0 else "overlay remains acyclic"
        index_entries.append({
            "href": page_name,
            "execution_unit_key": unit_dir.name,
            "iteration": str(summary.get("iteration", "")),
            "nodes": str(graph_meta.get("filtered_nodes", graph_meta.get("all_nodes", 0))),
            "edges": str(graph_meta.get("filtered_edges", graph_meta.get("all_edges", 0))),
            "topdown_mode": str(summary.get("topdown_mode", "")),
            "notes": notes,
        })

    index_title = f"{graph_prep_root.name} Cytoscape DAG Index"
    (output_dir / "index.html").write_text(render_index(index_title, index_entries), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cytoscape.js DAG pages from graph-prep bundles.")
    parser.add_argument("--graph-prep-root", required=True, help="Directory created by graph_join_preparation.py")
    parser.add_argument("--output-dir", required=True, help="Directory for generated Cytoscape HTML files")
    parser.add_argument("--mode", choices=["filtered", "all"], default="filtered", help="Choose filtered or all bundle inputs")
    parser.add_argument("--layout", choices=["dagre", "breadthfirst"], default="dagre", help="Use dagre when available, otherwise built-in breadthfirst")
    parser.add_argument("--cytoscape-url", default=DEFAULT_CYTOSCAPE_URL, help="Script URL for Cytoscape.js")
    parser.add_argument("--cytoscape-dagre-url", default=DEFAULT_CY_DAGRE_URL, help="Script URL for cytoscape-dagre")
    args = parser.parse_args()

    write_outputs(
        graph_prep_root=Path(args.graph_prep_root),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        use_dagre=args.layout == "dagre",
        cytoscape_url=args.cytoscape_url,
        cy_dagre_url=args.cytoscape_dagre_url,
    )


if __name__ == "__main__":
    main()
