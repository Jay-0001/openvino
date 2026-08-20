#!/usr/bin/env python3
"""
Clustered hierarchy visualizer for graph-prep bundles.

gsoc gtpin start
This supersedes the earlier radial mindmap layout with a more scalable view:
inference -> op type -> layer -> primitive -> kernel

The goal of this version is:
- robust zoom/pan
- a much larger world canvas
- better separation of dense layer sets through an intermediate op-type layer
- preservation of filtered vs unfiltered modes
gsoc gtpin end
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int(value: str, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(clamp(rgb[0], 0, 255)),
        int(clamp(rgb[1], 0, 255)),
        int(clamp(rgb[2], 0, 255)),
    )


def blend(color_a: str, color_b: str, t: float) -> str:
    ra, ga, ba = hex_to_rgb(color_a)
    rb, gb, bb = hex_to_rgb(color_b)
    return rgb_to_hex((lerp(ra, rb, t), lerp(ga, gb, t), lerp(ba, bb, t)))


def detect_bundle_prefix(unit_dir: Path) -> str:
    summary_files = sorted(unit_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary json found in {unit_dir}")
    return summary_files[0].name[:-len("_summary.json")]


def infer_model_name(graph_prep_root: Path) -> str:
    return graph_prep_root.name.replace("_", " ")


def compute_node_style(nodes: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    kernels = [node for node in nodes if node["node_type"] == "kernel"]
    values = [parse_int(node.get("total_cycles", "0")) for node in kernels]
    min_value = min(values) if values else 0
    max_value = max(values) if values else 0

    styles: Dict[str, Dict[str, object]] = {}
    for node in nodes:
        node_id = node["node_id"]
        node_type = node["node_type"]
        if node_type == "component":
            styles[node_id] = {"fill": "#5d7f78", "stroke": "#294740", "radius": 26}
        elif node_type == "op_type":
            styles[node_id] = {"fill": "#7f73c7", "stroke": "#43388b", "radius": 22}
        elif node_type == "origin_op":
            styles[node_id] = {"fill": "#d5a54c", "stroke": "#7a5917", "radius": 18}
        elif node_type == "primitive":
            mapped = node.get("mapping_status", "") == "mapped"
            styles[node_id] = {
                "fill": "#5d9d69" if mapped else "#b7b7b0",
                "stroke": "#1f4828" if mapped else "#6e6e68",
                "radius": 12 if mapped else 10,
            }
        elif node_type == "kernel":
            total_cycles = parse_int(node.get("total_cycles", "0"))
            if max_value > min_value:
                norm = (math.sqrt(max(total_cycles, 0)) - math.sqrt(max(min_value, 0))) / (
                    math.sqrt(max_value) - math.sqrt(max(min_value, 0))
                )
            else:
                norm = 0.0
            styles[node_id] = {
                "fill": blend("#74b9ff", "#ef6a44", norm),
                "stroke": blend("#2b4f7d", "#7d2817", norm),
                "radius": 8 + norm * 20,
            }
        else:
            styles[node_id] = {"fill": "#d7d7d7", "stroke": "#666666", "radius": 10}
    return styles


def compute_op_type_style(op_type_name: str, index: int) -> Dict[str, object]:
    palette = [
        ("#8267d0", "#49358f"),
        ("#5e9bd6", "#24547f"),
        ("#4ea981", "#1c5a3c"),
        ("#c38647", "#6f4616"),
        ("#c96f7f", "#7e3441"),
        ("#6f9f48", "#36581b"),
    ]
    fill, stroke = palette[index % len(palette)]
    return {"fill": fill, "stroke": stroke, "radius": 24}


def build_payload(unit_dir: Path, prefix: str, mode: str, model_name: str) -> Dict[str, object]:
    if mode == "filtered":
        nodes = read_csv(unit_dir / f"{prefix}_nodes_filtered.csv")
        edges = read_csv(unit_dir / f"{prefix}_edges_filtered.csv")
    else:
        nodes = read_csv(unit_dir / f"{prefix}_nodes_all.csv")
        edges = read_csv(unit_dir / f"{prefix}_edges_all.csv")

    summary = read_json(unit_dir / f"{prefix}_summary.json")
    base_styles = compute_node_style(nodes)
    nodes_by_id = {node["node_id"]: node for node in nodes}

    component_nodes = [node for node in nodes if node["node_type"] == "component"]
    op_type_nodes = [node for node in nodes if node["node_type"] == "op_type"]
    origin_nodes = [node for node in nodes if node["node_type"] == "origin_op"]
    layer_to_primitives: Dict[str, List[Dict[str, str]]] = {}
    primitive_to_kernels: Dict[str, List[Dict[str, str]]] = {}
    component_to_op_types: Dict[str, List[Dict[str, str]]] = {}
    op_type_to_layers: Dict[str, List[Dict[str, str]]] = {}
    dependency_edges = [edge for edge in edges if edge.get("edge_type") == "primitive_dependency"]

    for edge in edges:
        target = nodes_by_id.get(edge["target_node_id"])
        if not target:
            continue
        if edge.get("edge_type") == "component_to_op_type":
            component_to_op_types.setdefault(edge["source_node_id"], []).append(target)
        elif edge.get("edge_type") == "op_type_to_layer":
            op_type_to_layers.setdefault(edge["source_node_id"], []).append(target)
        elif edge.get("edge_type") == "layer_to_primitive":
            layer_to_primitives.setdefault(edge["source_node_id"], []).append(target)
        elif edge.get("edge_type") == "primitive_to_kernel":
            primitive_to_kernels.setdefault(edge["source_node_id"], []).append(target)

    ordered_components = sorted(
        component_nodes,
        key=lambda item: (-parse_int(item.get("layer_children", "0")), item.get("label", "")),
    )

    positioned_nodes: List[Dict[str, object]] = []
    node_positions: Dict[str, Tuple[float, float]] = {}
    rendered_edges: List[Dict[str, object]] = []

    x_root = 180.0
    x_component = 480.0
    x_op_type = 860.0
    x_layer = 1260.0
    x_primitive = 1740.0
    x_kernel = 2360.0

    center_y = 160.0
    current_y = 220.0
    component_gap = 190.0
    op_type_gap = 110.0
    layer_gap = 62.0
    primitive_gap = 30.0
    kernel_gap = 24.0

    root_id = f"{summary['execution_unit_key']}::inference_root"
    root_node = {
        "node_id": root_id,
        "node_type": "inference_root",
        "label": model_name,
        "sub_label": f"{summary['execution_unit_key']} | net {summary['net_id']} | iter {summary['iteration']}",
        "execution_unit_key": summary["execution_unit_key"],
        "net_id": summary["net_id"],
        "iteration": summary["iteration"],
        "fill": "#31435c",
        "stroke": "#14202c",
        "radius": 42,
        "x": x_root,
        "y": center_y,
    }
    positioned_nodes.append(root_node)
    node_positions[root_id] = (x_root, center_y)

    for component_index, component in enumerate(ordered_components):
        component_style = base_styles[component["node_id"]]
        op_types = sorted(
            component_to_op_types.get(component["node_id"], []),
            key=lambda item: (-parse_int(item.get("layer_children", "0")), item.get("label", "")),
        )

        group_units = 0
        for op_type in op_types:
            layers = sorted(op_type_to_layers.get(op_type["node_id"], []), key=lambda layer: layer.get("origin_op_name", ""))
            op_units = 0
            for layer in layers:
                primitives = sorted(layer_to_primitives.get(layer["node_id"], []), key=lambda item: item.get("primitive_id", ""))
                primitive_units = max(1, len(primitives))
                kernel_units = 0
                for primitive in primitives:
                    kernels = primitive_to_kernels.get(primitive["node_id"], [])
                    kernel_units += max(1, len(kernels))
                op_units += max(1, kernel_units, primitive_units)
            group_units += max(1, op_units)

        group_height = max(130.0, group_units * kernel_gap + len(op_types) * 36.0)
        component_y = current_y + group_height / 2.0
        positioned_nodes.append({
            **component,
            **component_style,
            "x": x_component,
            "y": component_y,
            "sub_label": component.get("component_group", ""),
        })
        node_positions[component["node_id"]] = (x_component, component_y)
        rendered_edges.append({
            "source": root_id,
            "target": component["node_id"],
            "edge_type": "inference_to_component",
            "style": "branch",
        })

        op_type_cursor_y = current_y
        for op_type_index, op_type in enumerate(op_types):
            op_style = compute_op_type_style(op_type.get("label", ""), component_index + op_type_index)
            layers = sorted(op_type_to_layers.get(op_type["node_id"], []), key=lambda layer: layer.get("origin_op_name", ""))
            op_units = 0
            for layer in layers:
                primitives = sorted(layer_to_primitives.get(layer["node_id"], []), key=lambda item: item.get("primitive_id", ""))
                primitive_units = max(1, len(primitives))
                kernel_units = 0
                for primitive in primitives:
                    kernels = primitive_to_kernels.get(primitive["node_id"], [])
                    kernel_units += max(1, len(kernels))
                op_units += max(1, kernel_units, primitive_units)
            op_height = max(90.0, op_units * kernel_gap + len(layers) * 24.0)
            op_center_y = op_type_cursor_y + op_height / 2.0

            positioned_nodes.append({
                **op_type,
                "fill": op_style["fill"],
                "stroke": op_style["stroke"],
                "radius": op_style["radius"],
                "x": x_op_type,
                "y": op_center_y,
                "sub_label": f"{len(layers)} layers",
            })
            node_positions[op_type["node_id"]] = (x_op_type, op_center_y)
            rendered_edges.append({
                "source": component["node_id"],
                "target": op_type["node_id"],
                "edge_type": "component_to_op_type",
                "style": "component",
            })

            layer_cursor_y = op_type_cursor_y
            for layer in layers:
                layer_style = base_styles[layer["node_id"]]
                primitives = sorted(layer_to_primitives.get(layer["node_id"], []), key=lambda item: item.get("primitive_id", ""))
                if not primitives:
                    layer_height = layer_gap
                else:
                    primitive_blocks = []
                    for primitive in primitives:
                        kernels = sorted(primitive_to_kernels.get(primitive["node_id"], []), key=lambda item: item.get("label", ""))
                        primitive_blocks.append(max(kernel_gap, len(kernels) * kernel_gap))
                    layer_height = max(layer_gap, sum(primitive_blocks) + max(0, len(primitives) - 1) * primitive_gap)

                layer_y = layer_cursor_y + layer_height / 2.0
                positioned_nodes.append({
                    **layer,
                    **layer_style,
                    "x": x_layer,
                    "y": layer_y,
                    "sub_label": layer.get("resolved_op_type_name", "") or layer.get("origin_op_type_name", ""),
                })
                node_positions[layer["node_id"]] = (x_layer, layer_y)
                rendered_edges.append({
                    "source": op_type["node_id"],
                    "target": layer["node_id"],
                    "edge_type": "op_type_to_layer",
                    "style": "op_type",
                })

                primitive_cursor_y = layer_cursor_y
                for primitive in primitives:
                    primitive_style = base_styles[primitive["node_id"]]
                    kernels = sorted(primitive_to_kernels.get(primitive["node_id"], []), key=lambda item: item.get("label", ""))
                    primitive_block_height = max(kernel_gap, len(kernels) * kernel_gap)
                    primitive_y = primitive_cursor_y + primitive_block_height / 2.0
                    positioned_nodes.append({
                        **primitive,
                        **primitive_style,
                        "x": x_primitive,
                        "y": primitive_y,
                        "sub_label": primitive.get("primitive_type", ""),
                    })
                    node_positions[primitive["node_id"]] = (x_primitive, primitive_y)
                    rendered_edges.append({
                        "source": layer["node_id"],
                        "target": primitive["node_id"],
                        "edge_type": "layer_to_primitive",
                        "style": "hierarchy",
                    })

                    kernel_cursor_y = primitive_cursor_y
                    for kernel in kernels:
                        kernel_style = base_styles[kernel["node_id"]]
                        kernel_y = kernel_cursor_y + kernel_gap / 2.0
                        positioned_nodes.append({
                            **kernel,
                            **kernel_style,
                            "x": x_kernel,
                            "y": kernel_y,
                            "sub_label": kernel.get("kernel_entries", ""),
                        })
                        node_positions[kernel["node_id"]] = (x_kernel, kernel_y)
                        rendered_edges.append({
                            "source": primitive["node_id"],
                            "target": kernel["node_id"],
                            "edge_type": "primitive_to_kernel",
                            "style": "kernel",
                        })
                        kernel_cursor_y += kernel_gap

                    primitive_cursor_y += primitive_block_height + primitive_gap

                layer_cursor_y += layer_height + layer_gap

            op_type_cursor_y += op_height + op_type_gap

        current_y += group_height + component_gap

    if mode == "unfiltered":
        for edge in dependency_edges:
            if edge["source_node_id"] in node_positions and edge["target_node_id"] in node_positions:
                rendered_edges.append({
                    "source": edge["source_node_id"],
                    "target": edge["target_node_id"],
                    "edge_type": "primitive_dependency",
                    "style": "dependency",
                })

    world_width = x_kernel + 420.0
    world_height = max(1600.0, current_y + 240.0)

    return {
        "mode": mode,
        "model_name": model_name,
        "summary": summary,
        "world": {"width": world_width, "height": world_height},
        "nodes": positioned_nodes,
        "edges": rendered_edges,
    }


def render_html(title: str, payload: Dict[str, object]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: linear-gradient(180deg, #f7f1e6 0%, #eee7d7 55%, #e4ddcf 100%);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: #171717;
    }}
    #app {{
      position: relative;
      width: 100%;
      height: 100%;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }}
    .overlay {{
      position: absolute;
      top: 18px;
      left: 18px;
      width: 360px;
      max-height: calc(100vh - 36px);
      overflow: auto;
      background: rgba(255, 252, 246, 0.93);
      border: 1px solid rgba(107, 89, 50, 0.22);
      box-shadow: 0 18px 40px rgba(50, 40, 18, 0.12);
      backdrop-filter: blur(12px);
      padding: 18px;
      box-sizing: border-box;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 24px;
      line-height: 1.15;
    }}
    h2 {{
      margin: 16px 0 8px;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #5d5138;
    }}
    .meta, .details, .legend {{
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.5;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 6px 0;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(0, 0, 0, 0.25);
      flex: 0 0 auto;
    }}
    .search {{
      width: 100%;
      box-sizing: border-box;
      padding: 10px 12px;
      border: 1px solid rgba(108, 92, 60, 0.32);
      background: white;
      font-size: 14px;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }}
    .actions button {{
      border: 1px solid rgba(108, 92, 60, 0.32);
      background: white;
      padding: 8px 10px;
      font-size: 13px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <div id="app">
    <canvas id="graph"></canvas>
    <div class="overlay">
      <h1>{title}</h1>
      <div class="meta" id="meta"></div>
      <h2>Search</h2>
      <input id="search" class="search" placeholder="Search op type, layer, primitive, kernel">
      <div class="actions">
        <button id="fit">Fit View</button>
        <button id="zoomIn">Zoom In</button>
        <button id="zoomOut">Zoom Out</button>
      </div>
      <h2>Legend</h2>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="background:#31435c"></span> inference root</div>
        <div class="legend-row"><span class="swatch" style="background:#5d7f78"></span> component cluster</div>
        <div class="legend-row"><span class="swatch" style="background:#8267d0"></span> op type cluster</div>
        <div class="legend-row"><span class="swatch" style="background:#d5a54c"></span> layer / origin op</div>
        <div class="legend-row"><span class="swatch" style="background:#5d9d69"></span> mapped primitive</div>
        <div class="legend-row"><span class="swatch" style="background:#b7b7b0"></span> topology-only primitive</div>
        <div class="legend-row"><span class="swatch" style="background:linear-gradient(90deg,#74b9ff,#ef6a44)"></span> kernel heat + size by total cycles</div>
      </div>
      <h2>Selected Node</h2>
      <div class="details" id="details">Click a node to inspect its joined row.</div>
    </div>
  </div>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    const canvas = document.getElementById('graph');
    const ctx = canvas.getContext('2d');
    const meta = document.getElementById('meta');
    const details = document.getElementById('details');
    const search = document.getElementById('search');
    const fitButton = document.getElementById('fit');
    const zoomInButton = document.getElementById('zoomIn');
    const zoomOutButton = document.getElementById('zoomOut');

    meta.textContent =
      `model=${{payload.model_name}}\\n` +
      `execution_unit=${{payload.summary.execution_unit_key}} net=${{payload.summary.net_id}} iter=${{payload.summary.iteration}}\\n` +
      `dispatch_rows=${{payload.summary.dispatch_rows_total}} mapped_primitives=${{payload.summary.topdown_primitives_mapped}}/${{payload.summary.topdown_primitives_total}}\\n` +
      `mode=${{payload.mode}} world=${{Math.round(payload.world.width)}}x${{Math.round(payload.world.height)}}`;

    const nodes = payload.nodes;
    const edges = payload.edges;
    const nodeMap = new Map(nodes.map((node) => [node.node_id, node]));

    let selectedNodeId = null;
    let filterText = '';
    let panX = 0;
    let panY = 0;
    let scale = 1;
    let dragging = false;
    let dragMoved = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let lastX = 0;
    let lastY = 0;

    function resizeCanvas() {{
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * ratio);
      canvas.height = Math.floor(window.innerHeight * ratio);
      canvas.style.width = `${{window.innerWidth}}px`;
      canvas.style.height = `${{window.innerHeight}}px`;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      fitToView();
      draw();
    }}

    function fitToView() {{
      const width = Math.max(1, payload.world.width);
      const height = Math.max(1, payload.world.height);
      const viewportW = window.innerWidth;
      const viewportH = window.innerHeight;
      const scaleX = viewportW / width;
      const scaleY = viewportH / height;
      scale = Math.min(scaleX, scaleY) * 0.92;
      panX = (viewportW - width * scale) / 2;
      panY = (viewportH - height * scale) / 2;
    }}

    function worldToScreen(x, y) {{
      return {{ x: x * scale + panX, y: y * scale + panY }};
    }}

    function screenToWorld(x, y) {{
      return {{ x: (x - panX) / scale, y: (y - panY) / scale }};
    }}

    function matchesFilter(node) {{
      if (!filterText) return true;
      const hay = [
        node.label || '',
        node.origin_op_name || '',
        node.origin_op_type_name || '',
        node.primitive_id || '',
        node.kernel_entries || '',
        node.sub_label || '',
        node.node_type || ''
      ].join(' ').toLowerCase();
      return hay.includes(filterText);
    }}

    function shouldShowDependency(edge) {{
      if (payload.mode !== 'unfiltered') return false;
      if (edge.edge_type !== 'primitive_dependency') return false;
      if (!selectedNodeId) return false;
      return edge.source === selectedNodeId || edge.target === selectedNodeId;
    }}

    function drawEdge(edge) {{
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return;
      if (!matchesFilter(source) && !matchesFilter(target)) return;

      const a = worldToScreen(source.x, source.y);
      const b = worldToScreen(target.x, target.y);
      const midX = (a.x + b.x) / 2;

      ctx.save();
      if (edge.style === 'branch') {{
        ctx.strokeStyle = 'rgba(108, 88, 49, 0.34)';
        ctx.lineWidth = Math.max(2.1, 2.8 * scale);
      }} else if (edge.style === 'component') {{
        ctx.strokeStyle = 'rgba(58, 92, 82, 0.30)';
        ctx.lineWidth = Math.max(1.7, 2.2 * scale);
      }} else if (edge.style === 'op_type') {{
        ctx.strokeStyle = 'rgba(93, 88, 133, 0.30)';
        ctx.lineWidth = Math.max(1.6, 2.0 * scale);
      }} else if (edge.style === 'hierarchy') {{
        ctx.strokeStyle = 'rgba(90, 98, 82, 0.22)';
        ctx.lineWidth = Math.max(1.1, 1.5 * scale);
      }} else if (edge.style === 'kernel') {{
        ctx.strokeStyle = 'rgba(70, 86, 120, 0.22)';
        ctx.lineWidth = Math.max(1.0, 1.4 * scale);
      }} else {{
        ctx.strokeStyle = 'rgba(177, 77, 28, 0.62)';
        ctx.lineWidth = Math.max(1.0, 1.3 * scale);
        ctx.setLineDash([6, 5]);
      }}
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.bezierCurveTo(midX, a.y, midX, b.y, b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }}

    function drawNode(node) {{
      if (!matchesFilter(node)) return;
      const p = worldToScreen(node.x, node.y);
      const radius = node.radius * scale;
      const isSelected = node.node_id === selectedNodeId;

      ctx.save();
      ctx.beginPath();
      ctx.fillStyle = node.fill;
      ctx.strokeStyle = isSelected ? '#f75f37' : node.stroke;
      ctx.lineWidth = isSelected ? Math.max(2.5, 2.8 * scale) : Math.max(1.0, 1.2 * scale);
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      const showLabel =
        node.node_type !== 'kernel' ||
        scale > 0.40 ||
        isSelected;

      if (showLabel) {{
        ctx.fillStyle = '#171717';
        ctx.textBaseline = 'middle';
        ctx.font = `${{Math.max(11, 12 * scale)}}px Trebuchet MS`;
        ctx.fillText(node.label, p.x + radius + 8, p.y);
        if (scale > 0.75 && node.sub_label && node.node_type !== 'kernel') {{
          ctx.font = `${{Math.max(10, 10 * scale)}}px Trebuchet MS`;
          ctx.fillStyle = 'rgba(23, 23, 23, 0.68)';
          ctx.fillText(node.sub_label, p.x + radius + 8, p.y + 13);
        }}
      }}
      ctx.restore();
    }}

    function drawColumnHeaders() {{
      const headers = [
        ['Inference', 180],
        ['Component', 480],
        ['Op Type', 860],
        ['Layer', 1260],
        ['Primitive', 1740],
        ['Kernel', 2360],
      ];
      ctx.save();
      ctx.fillStyle = 'rgba(70, 58, 34, 0.82)';
      ctx.font = 'bold 16px Trebuchet MS';
      headers.forEach(([label, x]) => {{
        const p = worldToScreen(x, 58);
        ctx.fillText(label, p.x - 38, p.y);
      }});
      ctx.restore();
    }}

    function draw() {{
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      drawColumnHeaders();
      const visibleEdges = edges.filter((edge) => edge.edge_type !== 'primitive_dependency' || shouldShowDependency(edge));
      visibleEdges.forEach(drawEdge);
      nodes.forEach(drawNode);
    }}

    function hitTest(clientX, clientY) {{
      const world = screenToWorld(clientX, clientY);
      for (let i = nodes.length - 1; i >= 0; i--) {{
        const node = nodes[i];
        if (!matchesFilter(node)) continue;
        const dx = world.x - node.x;
        const dy = world.y - node.y;
        if ((dx * dx + dy * dy) <= (node.radius * node.radius)) {{
          return node;
        }}
      }}
      return null;
    }}

    canvas.addEventListener('mousedown', (event) => {{
      dragging = true;
      dragMoved = false;
      dragStartX = event.clientX;
      dragStartY = event.clientY;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.style.cursor = 'grabbing';
    }});

    window.addEventListener('mouseup', () => {{
      dragging = false;
      canvas.style.cursor = 'grab';
    }});

    window.addEventListener('mousemove', (event) => {{
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      if (Math.abs(event.clientX - dragStartX) > 4 || Math.abs(event.clientY - dragStartY) > 4) {{
        dragMoved = true;
      }}
      panX += dx;
      panY += dy;
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});

    canvas.addEventListener('wheel', (event) => {{
      event.preventDefault();
      const mouseX = event.clientX;
      const mouseY = event.clientY;
      const before = screenToWorld(mouseX, mouseY);
      const zoomFactor = event.deltaY < 0 ? 1.10 : 0.90;
      scale = clamp(scale * zoomFactor, 0.06, 3.0);
      const afterX = before.x * scale + panX;
      const afterY = before.y * scale + panY;
      panX += mouseX - afterX;
      panY += mouseY - afterY;
      draw();
    }}, {{ passive: false }});

    canvas.addEventListener('click', (event) => {{
      if (dragMoved) return;
      const node = hitTest(event.clientX, event.clientY);
      if (node) {{
        selectedNodeId = node.node_id;
        details.textContent = JSON.stringify(node, null, 2);
      }} else {{
        selectedNodeId = null;
        details.textContent = 'Click a node to inspect its joined row.';
      }}
      draw();
    }});

    search.addEventListener('input', () => {{
      filterText = search.value.trim().toLowerCase();
      draw();
    }});

    fitButton.addEventListener('click', () => {{
      fitToView();
      draw();
    }});
    zoomInButton.addEventListener('click', () => {{
      scale = clamp(scale * 1.18, 0.06, 3.0);
      draw();
    }});
    zoomOutButton.addEventListener('click', () => {{
      scale = clamp(scale * 0.84, 0.06, 3.0);
      draw();
    }});

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def generate_unit(graph_prep_root: Path, unit_dir: Path, model_name: str) -> None:
    prefix = detect_bundle_prefix(unit_dir)
    filtered_payload = build_payload(unit_dir, prefix, "filtered", model_name)
    unfiltered_payload = build_payload(unit_dir, prefix, "unfiltered", model_name)

    filtered_title = f"Filtered Clustered Hierarchy - {model_name} - {filtered_payload['summary']['execution_unit_key']}"
    unfiltered_title = f"Unfiltered Clustered Hierarchy - {model_name} - {unfiltered_payload['summary']['execution_unit_key']}"

    write_file(
        unit_dir / f"{prefix}_filtered_mindmap.html",
        render_html(filtered_title, filtered_payload),
    )
    write_file(
        unit_dir / f"{prefix}_unfiltered_mindmap.html",
        render_html(unfiltered_title, unfiltered_payload),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clustered hierarchy HTML files from graph-prep bundles.")
    parser.add_argument("--graph-prep-root", required=True, help="Root directory containing n*_i* execution-unit folders")
    parser.add_argument("--execution-unit", default="", help="Optional single execution-unit folder name to render")
    parser.add_argument("--model-name", default="", help="Optional display name for the root inference node")
    args = parser.parse_args()

    root = Path(args.graph_prep_root)
    model_name = args.model_name.strip() or infer_model_name(root)

    if args.execution_unit:
        unit_dir = root / args.execution_unit
        if not unit_dir.is_dir():
            raise FileNotFoundError(f"Execution unit directory not found: {unit_dir}")
        generate_unit(root, unit_dir, model_name)
        return

    unit_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("n"))
    if not unit_dirs:
        raise FileNotFoundError(f"No execution-unit directories found under {root}")

    for unit_dir in unit_dirs:
        generate_unit(root, unit_dir, model_name)


if __name__ == "__main__":
    main()
