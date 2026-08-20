#!/usr/bin/env python3
"""
gsoc gtpin

Generate static HTML/JS graph views from per-execution-unit graph-prep bundles.

This first version prioritizes:
- data consistency
- inference isolation
- low dependency footprint

It intentionally avoids external JS libraries and produces two different specs:
1. Filtered hierarchy view:
   - explicit layered node-link layout
   - uses filtered nodes/edges only
2. Unfiltered topology explorer:
   - grouped force-lite / edge-on-demand explorer
   - uses all nodes/edges but only renders dependency edges on selection
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def escape_js_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


def parse_int(value: str, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def build_hierarchy_payload(nodes: List[Dict[str, str]], edges: List[Dict[str, str]], summary: Dict[str, object]) -> Dict[str, object]:
    by_type: Dict[str, List[Dict[str, str]]] = {"component_group": [], "op_type": [], "origin_op": [], "primitive": [], "kernel": []}
    for node in nodes:
        by_type.setdefault(node["node_type"], []).append(node)

    for node_type in by_type:
        by_type[node_type].sort(key=lambda item: (item.get("origin_op_name", ""), item.get("primitive_id", ""), item.get("label", "")))

    row_gap = 28
    component_x = 180
    op_type_x = 460
    layer_x = 780
    primitive_x = 1220
    kernel_x = 1680

    component_positions = {}
    op_type_positions = {}
    layer_positions = {}
    primitive_positions = {}
    kernel_positions = {}

    for idx, node in enumerate(by_type.get("component_group", [])):
        component_positions[node["node_id"]] = {"x": component_x, "y": 60 + idx * row_gap}

    for idx, node in enumerate(by_type.get("op_type", [])):
        op_type_positions[node["node_id"]] = {"x": op_type_x, "y": 60 + idx * row_gap}

    for idx, node in enumerate(by_type.get("origin_op", [])):
        layer_positions[node["node_id"]] = {"x": layer_x, "y": 60 + idx * row_gap}

    for idx, node in enumerate(by_type.get("primitive", [])):
        primitive_positions[node["node_id"]] = {"x": primitive_x, "y": 60 + idx * row_gap}

    for idx, node in enumerate(by_type.get("kernel", [])):
        kernel_positions[node["node_id"]] = {"x": kernel_x, "y": 60 + idx * row_gap}

    positions = {}
    positions.update(component_positions)
    positions.update(op_type_positions)
    positions.update(layer_positions)
    positions.update(primitive_positions)
    positions.update(kernel_positions)

    rendered_nodes = []
    for node in nodes:
        pos = positions[node["node_id"]]
        rendered_nodes.append({
            **node,
            "x": pos["x"],
            "y": pos["y"],
            "total_cycles_num": parse_int(node.get("total_cycles", "0")),
            "dispatch_rows_num": parse_int(node.get("dispatch_rows", "0")),
            "kernel_entry_count_num": parse_int(node.get("kernel_entry_count", "0")),
        })

    rendered_edges = []
    for edge in edges:
        source = positions.get(edge["source_node_id"])
        target = positions.get(edge["target_node_id"])
        if not source or not target:
            continue
        rendered_edges.append({
            **edge,
            "x1": source["x"],
            "y1": source["y"],
            "x2": target["x"],
            "y2": target["y"],
        })

    return {
        "summary": summary,
        "columns": [
            {"label": "Component Groups", "x": component_x},
            {"label": "Op Types", "x": op_type_x},
            {"label": "Origin Ops", "x": layer_x},
            {"label": "Primitives", "x": primitive_x},
            {"label": "Kernels", "x": kernel_x},
        ],
        "canvas": {
            "width": 1960,
            "height": max(900, 120 + max((node["y"] for node in rendered_nodes), default=0)),
        },
        "nodes": rendered_nodes,
        "edges": rendered_edges,
    }


def build_topology_payload(nodes: List[Dict[str, str]], edges: List[Dict[str, str]], summary: Dict[str, object]) -> Dict[str, object]:
    nodes_by_id = {node["node_id"]: node for node in nodes}

    component_nodes = [node for node in nodes if node["node_type"] == "component_group"]
    op_type_nodes = [node for node in nodes if node["node_type"] == "op_type"]
    layer_nodes = [node for node in nodes if node["node_type"] == "origin_op"]
    primitive_nodes = [node for node in nodes if node["node_type"] == "primitive"]
    kernel_nodes = [node for node in nodes if node["node_type"] == "kernel"]

    grouped = {
        "component_group": sorted(component_nodes, key=lambda item: item.get("label", "")),
        "op_type": sorted(op_type_nodes, key=lambda item: item.get("label", "")),
        "origin_op": sorted(layer_nodes, key=lambda item: item.get("origin_op_name", "")),
        "primitive": sorted(primitive_nodes, key=lambda item: item.get("primitive_id", "")),
        "kernel": sorted(kernel_nodes, key=lambda item: item.get("label", "")),
    }

    for node_type, items in grouped.items():
        for index, node in enumerate(items):
            if node_type == "component_group":
                x = 240
            elif node_type == "op_type":
                x = 520
            elif node_type == "origin_op":
                x = 800
            elif node_type == "primitive":
                x = 980
            else:
                x = 1500
            node["_x"] = x
            node["_y"] = 70 + index * 22

    rendered_edges = []
    for edge in edges:
        source = nodes_by_id.get(edge["source_node_id"])
        target = nodes_by_id.get(edge["target_node_id"])
        if not source or not target:
            continue
        rendered_edges.append({
            **edge,
            "source_type": source["node_type"],
            "target_type": target["node_type"],
        })

    return {
        "summary": summary,
        "nodes": nodes,
        "edges": rendered_edges,
    }


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def render_filtered_html(title: str, payload: Dict[str, object]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f4f1ea;
      color: #1f1f1f;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid #cfc8b8;
      padding: 24px 20px;
      background: linear-gradient(180deg, #eee6d5 0%, #f6f2e8 100%);
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      box-sizing: border-box;
    }}
    .main {{
      overflow: auto;
      padding: 24px;
      box-sizing: border-box;
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 12px;
    }}
    h2 {{
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 24px 0 8px;
    }}
    .meta {{
      font-size: 14px;
      line-height: 1.5;
    }}
    .canvas-wrap {{
      background: white;
      border: 1px solid #cfc8b8;
      box-shadow: 0 10px 30px rgba(70, 57, 22, 0.08);
      overflow: auto;
    }}
    svg {{
      display: block;
      background:
        linear-gradient(90deg, rgba(112,92,56,0.05) 1px, transparent 1px) 0 0 / 180px 180px,
        linear-gradient(180deg, rgba(112,92,56,0.04) 1px, transparent 1px) 0 0 / 180px 180px,
        #fffdfa;
    }}
    .edge {{
      stroke: rgba(88, 73, 41, 0.28);
      stroke-width: 1.1;
      fill: none;
    }}
    .node-label {{
      font-size: 11px;
      dominant-baseline: middle;
    }}
    .column-label {{
      font-size: 15px;
      font-weight: bold;
      fill: #6b5a33;
    }}
    .node-circle {{
      fill: #fffdfa;
      stroke: #6f5d36;
      stroke-width: 1.2;
    }}
    .node-layer .node-circle {{
      fill: #f5ecd8;
    }}
    .node-primitive .node-circle {{
      fill: #e8f1ea;
    }}
    .node-kernel .node-circle {{
      fill: #edf1f9;
    }}
    .details {{
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.5;
      background: rgba(255,255,255,0.65);
      border: 1px solid #d7cfbe;
      padding: 12px;
    }}
    .hint {{
      font-size: 13px;
      color: #5d5649;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <h1>{html.escape(title)}</h1>
      <div class="meta" id="meta"></div>
      <h2>Selected Node</h2>
      <div class="details" id="details">Click a node to inspect its graph row.</div>
      <h2>Notes</h2>
      <div class="hint">
        This filtered view only renders hierarchy edges and mapped hierarchy nodes.
        It is meant to preserve data consistency while staying readable enough for first-pass validation.
      </div>
    </aside>
    <main class="main">
      <div class="canvas-wrap">
        <svg id="graph" width="100%" height="100%"></svg>
      </div>
    </main>
  </div>
  <script>
    const payload = {escape_js_json(payload)};
    const meta = document.getElementById('meta');
    const details = document.getElementById('details');
    const svg = document.getElementById('graph');

    meta.textContent =
      `execution_unit_key=${{payload.summary.execution_unit_key}}\\n` +
      `net_id=${{payload.summary.net_id}} iteration=${{payload.summary.iteration}}\\n` +
      `dispatch_rows=${{payload.summary.dispatch_rows_total}} mapped_primitives=${{payload.summary.topdown_primitives_mapped}}\\n` +
      `filtered_nodes=${{payload.summary.filtered_nodes ?? payload.nodes.length}} filtered_edges=${{payload.summary.filtered_edges ?? payload.edges.length}}`;

    svg.setAttribute('viewBox', `0 0 ${{payload.canvas.width}} ${{payload.canvas.height}}`);
    svg.setAttribute('width', payload.canvas.width);
    svg.setAttribute('height', payload.canvas.height);

    function create(tag, attrs = {{}}, text = '') {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      if (text) el.textContent = text;
      return el;
    }}

    payload.columns.forEach((column) => {{
      svg.appendChild(create('text', {{
        x: column.x - 50,
        y: 28,
        class: 'column-label'
      }}, column.label));
    }});

    payload.edges.forEach((edge) => {{
      const midX = (edge.x1 + edge.x2) / 2;
      const path = `M ${{edge.x1}} ${{edge.y1}} C ${{midX}} ${{edge.y1}}, ${{midX}} ${{edge.y2}}, ${{edge.x2}} ${{edge.y2}}`;
      svg.appendChild(create('path', {{ d: path, class: 'edge' }}));
    }});

    payload.nodes.forEach((node) => {{
      const g = create('g', {{
        class: `node-${{node.node_type === 'origin_op' ? 'layer' : node.node_type}}`,
        transform: `translate(${{node.x}}, ${{node.y}})`,
        tabindex: '0',
        role: 'button'
      }});
      const circle = create('circle', {{ r: 7, class: 'node-circle' }});
      const label = create('text', {{ x: 14, y: 0, class: 'node-label' }}, node.label);
      g.appendChild(circle);
      g.appendChild(label);
      const showDetails = () => {{
        details.textContent = JSON.stringify(node, null, 2);
      }};
      g.addEventListener('click', showDetails);
      g.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter' || event.key === ' ') {{
          event.preventDefault();
          showDetails();
        }}
      }});
      svg.appendChild(g);
    }});
  </script>
</body>
</html>
"""


def render_unfiltered_html(title: str, payload: Dict[str, object]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background: #f7f4ef;
      color: #1b1b1b;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      padding: 22px;
      background: #fbf8f2;
      border-right: 1px solid #dad2c5;
      box-sizing: border-box;
      overflow: auto;
    }}
    .main {{
      padding: 18px;
      box-sizing: border-box;
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 24px;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      margin: 18px 0;
    }}
    input[type="search"] {{
      width: 100%;
      padding: 10px 12px;
      font-size: 14px;
      border: 1px solid #c8bfaf;
      background: white;
      box-sizing: border-box;
    }}
    .panel {{
      border: 1px solid #d9d2c4;
      background: white;
      padding: 12px;
      margin-top: 14px;
    }}
    .canvas-wrap {{
      border: 1px solid #d9d2c4;
      background: linear-gradient(180deg, #fffdfa 0%, #f8f3ea 100%);
      box-shadow: 0 8px 28px rgba(51, 39, 13, 0.07);
      overflow: auto;
    }}
    canvas {{
      display: block;
      cursor: grab;
    }}
    .meta, .details {{
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.45;
    }}
    .legend {{
      font-size: 13px;
      line-height: 1.5;
    }}
    .legend strong {{
      display: inline-block;
      min-width: 88px;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <h1>{html.escape(title)}</h1>
      <div class="meta" id="meta"></div>
      <div class="controls">
        <input id="search" type="search" placeholder="Search label, primitive, kernel">
      </div>
      <div class="panel">
        <div class="legend">
          <div><strong>All nodes:</strong> hierarchy + topology-only rows</div>
          <div><strong>All edges:</strong> hierarchy edges always visible, dependency edges only for the selected node</div>
        </div>
      </div>
      <div class="panel">
        <div class="details" id="details">Click a node to inspect it and reveal only its dependency edges.</div>
      </div>
    </aside>
    <main class="main">
      <div class="canvas-wrap">
        <canvas id="graph" width="1900" height="5200"></canvas>
      </div>
    </main>
  </div>
  <script>
    const payload = {escape_js_json(payload)};
    const canvas = document.getElementById('graph');
    const ctx = canvas.getContext('2d');
    const meta = document.getElementById('meta');
    const details = document.getElementById('details');
    const search = document.getElementById('search');

    meta.textContent =
      `execution_unit_key=${{payload.summary.execution_unit_key}}\\n` +
      `net_id=${{payload.summary.net_id}} iteration=${{payload.summary.iteration}}\\n` +
      `all_nodes=${{payload.summary.all_nodes ?? payload.nodes.length}} all_edges=${{payload.summary.all_edges ?? payload.edges.length}}\\n` +
      `mapped_primitives=${{payload.summary.topdown_primitives_mapped}} topology_total=${{payload.summary.topdown_primitives_total}}`;

    const columns = {{
      origin_op: 220,
      primitive: 860,
      kernel: 1520
    }};

    const nodes = payload.nodes.map((node) => ({{ ...node }}));
    const byType = {{
      origin_op: nodes.filter((node) => node.node_type === 'origin_op'),
      primitive: nodes.filter((node) => node.node_type === 'primitive'),
      kernel: nodes.filter((node) => node.node_type === 'kernel')
    }};
    Object.entries(byType).forEach(([type, items]) => {{
      items.sort((a, b) => (a.label || '').localeCompare(b.label || ''));
      items.forEach((node, index) => {{
        node.x = columns[type];
        node.y = 70 + index * (type === 'kernel' ? 18 : 22);
      }});
    }});

    const nodeIndex = new Map(nodes.map((node) => [node.node_id, node]));
    const hierarchyEdges = payload.edges.filter((edge) => edge.edge_type !== 'primitive_dependency');
    const dependencyEdges = payload.edges.filter((edge) => edge.edge_type === 'primitive_dependency');

    let selectedNodeId = null;
    let searchText = '';

    function nodeMatchesSearch(node) {{
      if (!searchText) return true;
      const hay = [
        node.label,
        node.origin_op_name,
        node.primitive_id,
        node.kernel_entries
      ].join(' ').toLowerCase();
      return hay.includes(searchText);
    }}

    function visibleDependencyEdges() {{
      if (!selectedNodeId) return [];
      return dependencyEdges.filter((edge) => edge.source_node_id === selectedNodeId || edge.target_node_id === selectedNodeId);
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = '#6b5b3a';
      ctx.font = 'bold 16px Trebuchet MS';
      ctx.fillText('Origin Ops', 140, 28);
      ctx.fillText('Primitives', 790, 28);
      ctx.fillText('Kernels', 1450, 28);

      hierarchyEdges.forEach((edge) => {{
        const source = nodeIndex.get(edge.source_node_id);
        const target = nodeIndex.get(edge.target_node_id);
        if (!source || !target) return;
        if (!nodeMatchesSearch(source) && !nodeMatchesSearch(target)) return;
        ctx.strokeStyle = 'rgba(96, 82, 46, 0.14)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.bezierCurveTo((source.x + target.x) / 2, source.y, (source.x + target.x) / 2, target.y, target.x, target.y);
        ctx.stroke();
      }});

      visibleDependencyEdges().forEach((edge) => {{
        const source = nodeIndex.get(edge.source_node_id);
        const target = nodeIndex.get(edge.target_node_id);
        if (!source || !target) return;
        ctx.strokeStyle = 'rgba(160, 84, 44, 0.55)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.bezierCurveTo((source.x + target.x) / 2, source.y, (source.x + target.x) / 2, target.y, target.x, target.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }});

      nodes.forEach((node) => {{
        if (!nodeMatchesSearch(node)) return;
        let fill = '#fffdfa';
        if (node.node_type === 'origin_op') fill = '#efe3c7';
        if (node.node_type === 'primitive') fill = node.mapping_status === 'mapped' ? '#dfeee0' : '#f0f0ec';
        if (node.node_type === 'kernel') fill = '#e5ecf9';
        const isSelected = node.node_id === selectedNodeId;
        ctx.fillStyle = fill;
        ctx.strokeStyle = isSelected ? '#b14d1c' : '#6d603c';
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.node_type === 'kernel' ? 4.5 : 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#202020';
        ctx.font = '11px Trebuchet MS';
        ctx.fillText(node.label, node.x + 10, node.y + 4);
      }});
    }}

    function hitTest(x, y) {{
      for (const node of nodes) {{
        if (!nodeMatchesSearch(node)) continue;
        const radius = node.node_type === 'kernel' ? 6 : 8;
        const dx = x - node.x;
        const dy = y - node.y;
        if ((dx * dx + dy * dy) <= radius * radius) {{
          return node;
        }}
      }}
      return null;
    }}

    canvas.addEventListener('click', (event) => {{
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const node = hitTest(x, y);
      if (node) {{
        selectedNodeId = node.node_id;
        details.textContent = JSON.stringify(node, null, 2);
      }} else {{
        selectedNodeId = null;
        details.textContent = 'Click a node to inspect it and reveal only its dependency edges.';
      }}
      draw();
    }});

    search.addEventListener('input', () => {{
      searchText = search.value.trim().toLowerCase();
      draw();
    }});

    draw();
  </script>
</body>
</html>
"""


def detect_bundle_prefix(unit_dir: Path) -> str:
    summary_files = sorted(unit_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary json found in {unit_dir}")
    name = summary_files[0].name
    return name[:-len("_summary.json")]


def generate_for_unit(unit_dir: Path) -> None:
    prefix = detect_bundle_prefix(unit_dir)
    summary = read_json(unit_dir / f"{prefix}_summary.json")

    filtered_nodes = read_csv(unit_dir / f"{prefix}_nodes_filtered.csv")
    filtered_edges = read_csv(unit_dir / f"{prefix}_edges_filtered.csv")
    all_nodes = read_csv(unit_dir / f"{prefix}_nodes_all.csv")
    all_edges = read_csv(unit_dir / f"{prefix}_edges_all.csv")

    filtered_payload = build_hierarchy_payload(filtered_nodes, filtered_edges, summary)
    unfiltered_payload = build_topology_payload(all_nodes, all_edges, summary)

    title_base = f"{summary['execution_unit_key']} ({summary['net_id']} / iter {summary['iteration']})"
    write_file(
        unit_dir / f"{prefix}_filtered_graph.html",
        render_filtered_html(f"Filtered Hierarchy Graph - {title_base}", filtered_payload),
    )
    write_file(
        unit_dir / f"{prefix}_unfiltered_graph.html",
        render_unfiltered_html(f"Unfiltered Topology Explorer - {title_base}", unfiltered_payload),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static graph visualization HTML files from graph-prep bundles.")
    parser.add_argument("--graph-prep-root", required=True, help="Directory containing execution-unit subfolders such as n1_i0")
    parser.add_argument("--execution-unit", default="", help="Optional single execution unit folder name to render")
    args = parser.parse_args()

    root = Path(args.graph_prep_root)
    if args.execution_unit:
        unit_dir = root / args.execution_unit
        if not unit_dir.is_dir():
            raise FileNotFoundError(f"Execution unit folder not found: {unit_dir}")
        generate_for_unit(unit_dir)
        return

    unit_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("n"))
    if not unit_dirs:
        raise FileNotFoundError(f"No execution unit directories found under {root}")

    for unit_dir in unit_dirs:
        generate_for_unit(unit_dir)


if __name__ == "__main__":
    main()
