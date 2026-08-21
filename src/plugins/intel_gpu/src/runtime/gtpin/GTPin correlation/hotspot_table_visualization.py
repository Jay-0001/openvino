#!/usr/bin/env python3
"""
Generate static hotspot-analysis HTML views from hotspot-first graph-prep bundles.

The visualization is intentionally table-first:
- rank hotspots by total execution cycles
- expose multiple grouped views over the same rows
- expand rows inline instead of rendering a full topology graph
- keep drilldown compact and visual
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple


VIEW_DEFINITIONS = [
    {
        "id": "hierarchy",
        "label": "Hierarchy",
        "description": "Component path to kernel hotspot ranking.",
        "group_keys": ["component_path", "resolved_op_type_name", "origin_op_name", "kernel_entry"],
    },
    {
        "id": "primitive_type",
        "label": "Primitive Type",
        "description": "Primitive type to implementation to kernel hotspot ranking.",
        "group_keys": ["primitive_type", "implementation", "kernel_entry"],
    },
    {
        "id": "kernel_family",
        "label": "Kernel Family",
        "description": "Kernel family with attached hierarchy context.",
        "group_keys": ["kernel_entry", "component_path", "origin_op_name"],
    },
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: object, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def detect_bundle_prefix(graph_prep_root: Path) -> str:
    summaries = sorted(graph_prep_root.glob("*_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No bundle summary found in {graph_prep_root}")
    return summaries[0].name[: -len("_summary.json")]


def slugify(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return lowered or "view"


def unit_sort_key(row: Dict[str, str]) -> Tuple[int, int, str]:
    net_id = to_int(row.get("net_id", ""), default=999999)
    iteration = to_int(row.get("iteration", ""), default=999999)
    return net_id, iteration, str(row.get("execution_unit_key", ""))


def build_unit_metadata(row: Dict[str, str], hotspot_rows: List[Dict[str, str]]) -> Dict[str, object]:
    total_cycles = sum(to_int(item.get("gtpin_total_execution_cycles", "0")) for item in hotspot_rows)
    total_invocations = sum(to_int(item.get("gtpin_invocation_count", "0")) for item in hotspot_rows)
    avg_cycles = int(total_cycles / total_invocations) if total_invocations else 0
    return {
        "execution_unit_key": row.get("execution_unit_key", ""),
        "logical_execution_index": row.get("logical_execution_index", ""),
        "net_id": row.get("net_id", ""),
        "iteration": row.get("iteration", ""),
        "is_internal_network_candidate": row.get("is_internal_network_candidate", ""),
        "dispatch_rows_total": to_int(row.get("dispatch_rows_total", "0")),
        "matched_dispatch_rows": to_int(row.get("matched_dispatch_rows", "0")),
        "kernel_aligned_rows": to_int(row.get("kernel_aligned_rows", "0")),
        "topdown_primitives_total": to_int(row.get("topdown_primitives_total", "0")),
        "topdown_primitives_mapped": to_int(row.get("topdown_primitives_mapped", "0")),
        "topdown_mode": row.get("topdown_mode", ""),
        "hotspot_rows": len(hotspot_rows),
        "total_cycles": total_cycles,
        "total_invocations": total_invocations,
        "avg_cycles_per_invocation": avg_cycles,
        "distinct_components": len({row.get("component_path", "") for row in hotspot_rows}),
        "distinct_ops": len({row.get("origin_op_name", "") for row in hotspot_rows}),
        "distinct_kernels": len({row.get("kernel_entry", "") for row in hotspot_rows}),
    }


def summarize_all_units(unit_rows: List[Dict[str, str]], hotspots_by_unit: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for row in sorted(unit_rows, key=unit_sort_key):
        unit_key = str(row.get("execution_unit_key", ""))
        items.append(build_unit_metadata(row, hotspots_by_unit.get(unit_key, [])))
    return items


def build_distribution_chart(unit_summaries: List[Dict[str, object]]) -> str:
    steady_units = [item for item in unit_summaries if str(item.get("is_internal_network_candidate", "")).lower() != "true"]
    if not steady_units:
        return "<div class=\"chart-empty\">No steady inference runs available for deviation chart.</div>"

    width = 1080
    height = 320
    margin_left = 88
    margin_right = 34
    margin_top = 26
    margin_bottom = 54
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    count = max(len(steady_units), 1)
    step_x = inner_width / max(count - 1, 1)
    max_cycles = max(int(item["total_cycles"]) for item in steady_units)
    max_avg = max(int(item["avg_cycles_per_invocation"]) for item in steady_units)
    mean_cycles = sum(int(item["total_cycles"]) for item in steady_units) / count
    mean_avg = sum(int(item["avg_cycles_per_invocation"]) for item in steady_units) / count

    def x_at(index: int) -> float:
        if count == 1:
            return margin_left + inner_width / 2
        return margin_left + index * step_x

    def y_cycles(value: int) -> float:
        if max_cycles <= 0:
            return margin_top + inner_height
        return margin_top + inner_height - (value / max_cycles) * inner_height

    def y_avg(value: int) -> float:
        if max_avg <= 0:
            return margin_top + inner_height
        return margin_top + inner_height - (value / max_avg) * inner_height

    cycle_points = " ".join(f"{x_at(i):.1f},{y_cycles(int(item['total_cycles'])):.1f}" for i, item in enumerate(steady_units))
    avg_points = " ".join(f"{x_at(i):.1f},{y_avg(int(item['avg_cycles_per_invocation'])):.1f}" for i, item in enumerate(steady_units))
    x_labels = []
    for i, item in enumerate(steady_units):
        x = x_at(i)
        label = escape(str(item["execution_unit_key"]))
        x_labels.append(f"<text x=\"{x:.1f}\" y=\"{height - 18}\" text-anchor=\"middle\">{label}</text>")

    y_ticks = []
    for tick in range(5):
        ratio = tick / 4
        y = margin_top + inner_height - ratio * inner_height
        cycle_value = int(max_cycles * ratio)
        y_ticks.append(
            f"<line x1=\"{margin_left}\" y1=\"{y:.1f}\" x2=\"{width - margin_right}\" y2=\"{y:.1f}\" class=\"grid-line\" />"
            f"<text x=\"{margin_left - 12}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">{cycle_value // 1_000_000:,}M</text>"
        )

    cycle_mean_y = y_cycles(int(mean_cycles))
    avg_mean_y = y_avg(int(mean_avg))
    dots = []
    for i, item in enumerate(steady_units):
        x = x_at(i)
        cycle_val = int(item["total_cycles"])
        avg_val = int(item["avg_cycles_per_invocation"])
        dots.append(
            f"<circle cx=\"{x:.1f}\" cy=\"{y_cycles(cycle_val):.1f}\" r=\"4.5\" fill=\"#ff9a63\" />"
            f"<circle cx=\"{x:.1f}\" cy=\"{y_avg(avg_val):.1f}\" r=\"4.5\" fill=\"#61c2ff\" />"
        )

    return f"""
    <div class="chart-shell">
      <div class="chart-legend">
        <span><i class="legend-swatch cycles"></i>Total cycles</span>
        <span><i class="legend-swatch avg"></i>Avg cycles / invocation</span>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Deviation chart across all steady inference runs">
        <rect x="{margin_left}" y="{margin_top}" width="{inner_width}" height="{inner_height}" class="chart-frame"></rect>
        {''.join(y_ticks)}
        <line x1="{margin_left}" y1="{cycle_mean_y:.1f}" x2="{width - margin_right}" y2="{cycle_mean_y:.1f}" class="mean-line cycles" />
        <line x1="{margin_left}" y1="{avg_mean_y:.1f}" x2="{width - margin_right}" y2="{avg_mean_y:.1f}" class="mean-line avg" />
        <polyline points="{cycle_points}" class="series-line cycles" />
        <polyline points="{avg_points}" class="series-line avg" />
        {''.join(dots)}
        {''.join(x_labels)}
        <text x="{margin_left}" y="18" class="chart-title">Deviation across all steady inference runs</text>
        <text x="{margin_left}" y="{height - 6}" class="axis-label">Execution unit</text>
        <text transform="translate(18 {margin_top + inner_height / 2:.1f}) rotate(-90)" class="axis-label">Relative metric scale</text>
      </svg>
    </div>
    """


def render_index_page(
    output_dir: Path,
    title: str,
    prefix: str,
    unit_summaries: List[Dict[str, object]],
) -> None:
    steady_summaries = [item for item in unit_summaries if str(item.get("is_internal_network_candidate", "")).lower() != "true"]
    visible_summaries = steady_summaries[:5] if steady_summaries else unit_summaries[:5]
    rows_html: List[str] = []
    max_cycles = max((int(item["total_cycles"]) for item in visible_summaries), default=0)
    for summary in visible_summaries:
        unit_key = str(summary["execution_unit_key"])
        page_name = f"{prefix}_{unit_key}_hotspots.html"
        intensity = 0.0
        if max_cycles > 0:
            intensity = int(summary["total_cycles"]) / max_cycles
        hue = int(200 - intensity * 170)
        rows_html.append(
            "<tr>"
            f"<td><a href=\"{escape(page_name)}\">{escape(unit_key)}</a></td>"
            f"<td>{escape(str(summary['iteration']))}</td>"
            f"<td>{escape(str(summary['logical_execution_index']))}</td>"
            f"<td>{escape(str(summary['hotspot_rows']))}</td>"
            f"<td>{escape(str(summary['distinct_kernels']))}</td>"
            f"<td>{escape(str(summary['distinct_ops']))}</td>"
            f"<td class=\"metric-cell\"><span>{int(summary['total_cycles']):,}</span><div class=\"mini-bar\"><span style=\"width:{intensity * 100:.1f}%; background:hsl({hue} 82% 46%)\"></span></div></td>"
            f"<td>{int(summary['avg_cycles_per_invocation']):,}</td>"
            f"<td>{escape(str(summary['topdown_primitives_mapped']))} / {escape(str(summary['topdown_primitives_total']))}</td>"
            "</tr>"
        )

    chart_html = build_distribution_chart(unit_summaries)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0c1118;
      --panel: #121a23;
      --panel-2: #172230;
      --border: #263447;
      --text: #e9eef5;
      --muted: #91a1b5;
      --accent: #53b3ff;
      --warm: #ff8e53;
      --good: #6ad48c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(180deg, #091019 0%, var(--bg) 30%, #0f1822 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 36px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      font-weight: 600;
    }}
    p {{
      margin: 0 0 18px;
      color: var(--muted);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .stat {{
      background: rgba(18, 26, 35, 0.82);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat-value {{
      font-size: 26px;
      font-weight: 650;
    }}
    .table-shell {{
      background: rgba(18, 26, 35, 0.82);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(38, 52, 71, 0.85);
      vertical-align: top;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      background: rgba(23, 34, 48, 0.85);
      font-weight: 600;
    }}
    tr:hover td {{
      background: rgba(83, 179, 255, 0.04);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .metric-cell {{
      min-width: 220px;
    }}
    .mini-bar {{
      margin-top: 6px;
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      overflow: hidden;
    }}
    .mini-bar > span {{
      display: block;
      height: 100%;
      border-radius: inherit;
    }}
    .chart-card {{
      background: rgba(18, 26, 35, 0.82);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
    }}
    .chart-card h2 {{
      margin: 0 0 6px;
      font-size: 19px;
      font-weight: 620;
    }}
    .chart-card p {{
      margin: 0 0 12px;
      font-size: 13px;
    }}
    .chart-shell {{
      width: 100%;
      overflow-x: auto;
    }}
    .chart-shell svg {{
      width: 100%;
      min-width: 920px;
      height: auto;
      display: block;
    }}
    .chart-frame {{
      fill: rgba(255,255,255,0.02);
      stroke: rgba(255,255,255,0.08);
    }}
    .grid-line {{
      stroke: rgba(255,255,255,0.08);
      stroke-width: 1;
    }}
    .series-line {{
      fill: none;
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .series-line.cycles {{
      stroke: #ff9a63;
    }}
    .series-line.avg {{
      stroke: #61c2ff;
    }}
    .mean-line {{
      stroke-width: 1.5;
      stroke-dasharray: 5 5;
      opacity: 0.9;
    }}
    .mean-line.cycles {{
      stroke: rgba(255, 154, 99, 0.7);
    }}
    .mean-line.avg {{
      stroke: rgba(97, 194, 255, 0.7);
    }}
    .chart-title {{
      fill: var(--text);
      font-size: 16px;
      font-weight: 600;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 12px;
    }}
    svg text {{
      fill: var(--muted);
      font-size: 11px;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    .chart-legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .chart-legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .legend-swatch.cycles {{
      background: #ff9a63;
    }}
    .legend-swatch.avg {{
      background: #61c2ff;
    }}
    .chart-empty {{
      color: var(--muted);
      padding: 12px 0 4px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{escape(title)}</h1>
    <p>Hotspot-first execution-unit index. The table below focuses on the first five steady inference runs, while the deviation chart captures the full spread across all steady runs.</p>
    <div class="summary-grid">
      <div class="stat"><span class="stat-label">Execution Units</span><span class="stat-value">{len(unit_summaries)}</span></div>
      <div class="stat"><span class="stat-label">Hotspot Rows</span><span class="stat-value">{sum(int(item['hotspot_rows']) for item in unit_summaries):,}</span></div>
      <div class="stat"><span class="stat-label">Distinct Kernels</span><span class="stat-value">{sum(int(item['distinct_kernels']) for item in unit_summaries):,}</span></div>
      <div class="stat"><span class="stat-label">Total Cycles</span><span class="stat-value">{sum(int(item['total_cycles']) for item in unit_summaries):,}</span></div>
    </div>
    <div class="table-shell">
      <table>
        <thead>
          <tr>
            <th>Execution Unit</th>
            <th>Iteration</th>
            <th>Logical Index</th>
            <th>Hotspot Rows</th>
            <th>Distinct Kernels</th>
            <th>Distinct Ops</th>
            <th>Total Cycles</th>
            <th>Avg Cycles / Invocation</th>
            <th>Mapped Primitives</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    <div class="chart-card">
      <h2>Run Deviation</h2>
      <p>Total cycles and average cycles per invocation across all steady inference runs.</p>
      {chart_html}
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def render_unit_page(
    output_dir: Path,
    prefix: str,
    summary: Dict[str, object],
    hotspot_rows: List[Dict[str, str]],
) -> None:
    page_name = f"{prefix}_{summary['execution_unit_key']}_hotspots.html"
    page_title = f"Hotspot Analysis - {summary['execution_unit_key']}"
    payload = {
        "summary": summary,
        "views": VIEW_DEFINITIONS,
        "rows": hotspot_rows,
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1118;
      --bg-2: #0e1721;
      --panel: rgba(18, 26, 35, 0.88);
      --panel-strong: rgba(23, 34, 48, 0.95);
      --border: #253446;
      --text: #e8edf5;
      --muted: #8ea0b7;
      --accent: #59b7ff;
      --accent-2: #6de1b4;
      --hot: #ff8c52;
      --warn: #ffd166;
      --ok: #67d38b;
      --shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(89, 183, 255, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 140, 82, 0.12), transparent 28%),
        linear-gradient(180deg, #081018 0%, var(--bg) 45%, var(--bg-2) 100%);
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px 18px 36px;
    }}
    .topbar {{
      display: flex;
      gap: 12px;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .heading h1 {{
      margin: 0 0 6px;
      font-size: 32px;
      font-weight: 650;
    }}
    .heading p {{
      margin: 0;
      color: var(--muted);
      max-width: 880px;
      line-height: 1.45;
    }}
    .backlink {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat {{
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: var(--shadow);
      min-height: 96px;
    }}
    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .stat-value {{
      font-size: 26px;
      font-weight: 670;
    }}
    .stat-sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.9fr) minmax(320px, 0.95fr);
      gap: 18px;
      align-items: start;
    }}
    .left-column, .right-column {{
      display: grid;
      gap: 18px;
    }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-head {{
      padding: 14px 16px 12px;
      border-bottom: 1px solid rgba(37, 52, 70, 0.92);
      background: linear-gradient(180deg, rgba(23, 34, 48, 0.95), rgba(18, 26, 35, 0.98));
    }}
    .panel-head h2 {{
      margin: 0 0 6px;
      font-size: 17px;
      font-weight: 650;
    }}
    .panel-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid rgba(37, 52, 70, 0.92);
      background: rgba(11, 17, 24, 0.38);
    }}
    .view-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .view-tab {{
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.02);
      color: var(--muted);
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }}
    .view-tab.active {{
      color: var(--text);
      background: rgba(89, 183, 255, 0.15);
      border-color: rgba(89, 183, 255, 0.45);
    }}
    .control-group {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-left: auto;
    }}
    .control-label {{
      color: var(--muted);
      font-size: 13px;
    }}
    select {{
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }}
    th, td {{
      border-bottom: 1px solid rgba(37, 52, 70, 0.82);
      padding: 11px 12px;
      vertical-align: middle;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(23, 34, 48, 0.98);
      color: var(--muted);
      font-weight: 600;
    }}
    tbody tr {{
      transition: background 140ms ease;
    }}
    tbody tr:hover td {{
      background: rgba(89, 183, 255, 0.035);
    }}
    tbody tr.selected td {{
      background: rgba(89, 183, 255, 0.08);
    }}
    .row-label {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }}
    .expand-btn {{
      width: 28px;
      height: 28px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      flex: 0 0 auto;
    }}
    .expand-btn.disabled {{
      visibility: hidden;
      pointer-events: none;
    }}
    .label-stack {{
      min-width: 0;
    }}
    .label-main {{
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 520px;
    }}
    .label-sub {{
      margin-top: 3px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 560px;
    }}
    .heat-cell {{
      min-width: 220px;
    }}
    .heat-metric {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .heat-bar {{
      flex: 1 1 auto;
      height: 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      overflow: hidden;
    }}
    .heat-fill {{
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #4bb7ff 0%, #7adfba 42%, #ffd166 75%, #ff8c52 100%);
    }}
    .metric-note {{
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .intensity-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: 0 0 auto;
      box-shadow: 0 0 18px rgba(255, 140, 82, 0.22);
    }}
    .detail-body {{
      padding: 16px;
      display: grid;
      gap: 16px;
    }}
    .placeholder {{
      color: var(--muted);
      line-height: 1.5;
    }}
    .crumbs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .crumb {{
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(89, 183, 255, 0.10);
      color: var(--text);
      font-size: 12px;
      border: 1px solid rgba(89, 183, 255, 0.18);
    }}
    .detail-title {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      line-height: 1.35;
    }}
    .detail-sub {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .detail-stat {{
      padding: 12px;
      border-radius: 14px;
      border: 1px solid rgba(37, 52, 70, 0.92);
      background: rgba(12, 19, 28, 0.72);
    }}
    .detail-stat-label {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .detail-stat-value {{
      font-size: 20px;
      font-weight: 650;
    }}
    .quality-list {{
      display: grid;
      gap: 8px;
    }}
    .quality-item {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(12, 19, 28, 0.72);
      border: 1px solid rgba(37, 52, 70, 0.92);
      font-size: 13px;
    }}
    .quality-item strong {{
      color: var(--muted);
      font-weight: 600;
    }}
    .quality-ok {{
      color: var(--ok);
      font-weight: 600;
    }}
    .quality-warn {{
      color: var(--warn);
      font-weight: 600;
    }}
    .peer-bar-shell {{
      display: grid;
      gap: 8px;
    }}
    .peer-bar {{
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.06);
    }}
    .peer-bar > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #57b7ff 0%, #7de1b3 48%, #ff9358 100%);
      border-radius: inherit;
    }}
    .mono {{
      font-family: Consolas, "Courier New", monospace;
      word-break: break-word;
    }}
    .small-list {{
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .small-list strong {{
      color: var(--text);
    }}
    .kernel-list {{
      display: grid;
      gap: 8px;
    }}
    .kernel-chip {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(12, 19, 28, 0.72);
      border: 1px solid rgba(37, 52, 70, 0.92);
      font-size: 13px;
    }}
    .kernel-chip .mono {{
      color: var(--text);
      max-width: 78%;
    }}
    .kernel-chip-metric {{
      color: var(--warn);
      font-weight: 600;
      white-space: nowrap;
    }}
    @media (max-width: 1180px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .wrap {{
        padding: 18px 14px 28px;
      }}
      .heading h1 {{
        font-size: 26px;
      }}
      .detail-grid {{
        grid-template-columns: 1fr;
      }}
      .control-group {{
        width: 100%;
        margin-left: 0;
      }}
      .control-group select {{
        flex: 1 1 auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="heading">
        <h1>{escape(page_title)}</h1>
        <p>Ranked hotspot analysis using total execution cycles as the primary signal. Expand grouped rows inline, then inspect a selected hotspot through a compact detail panel.</p>
      </div>
      <a class="backlink" href="index.html">Back to execution-unit index</a>
    </div>

    <div class="stats">
      <div class="stat"><span class="stat-label">Execution Unit</span><div class="stat-value">{escape(str(summary["execution_unit_key"]))}</div><div class="stat-sub">net {escape(str(summary["net_id"]))}, iteration {escape(str(summary["iteration"]))}</div></div>
      <div class="stat"><span class="stat-label">Total Cycles</span><div class="stat-value">{int(summary["total_cycles"]):,}</div><div class="stat-sub">{int(summary["avg_cycles_per_invocation"]):,} avg cycles / invocation</div></div>
      <div class="stat"><span class="stat-label">Hotspot Rows</span><div class="stat-value">{int(summary["hotspot_rows"]):,}</div><div class="stat-sub">{int(summary["distinct_kernels"]):,} distinct kernels</div></div>
      <div class="stat"><span class="stat-label">Mapped Primitives</span><div class="stat-value">{int(summary["topdown_primitives_mapped"]):,}</div><div class="stat-sub">{int(summary["topdown_primitives_total"]):,} matched primitive rows available</div></div>
      <div class="stat"><span class="stat-label">Dispatch Coverage</span><div class="stat-value">{int(summary["matched_dispatch_rows"]):,}</div><div class="stat-sub">{int(summary["dispatch_rows_total"]):,} dispatch rows in unit</div></div>
    </div>

    <div class="layout">
      <div class="left-column">
        <div class="panel">
          <div class="panel-head">
            <h2>Hotspot Views</h2>
            <p>Switch grouping modes to inspect the same hotspot data from hierarchy, implementation, primitive type, or kernel-family angles.</p>
          </div>
          <div class="controls">
            <div class="view-tabs" id="view-tabs"></div>
            <div class="control-group">
              <span class="control-label">Rows</span>
              <select id="row-limit">
                <option value="25">Top 25</option>
                <option value="50" selected>Top 50</option>
                <option value="100">Top 100</option>
                <option value="250">Top 250</option>
                <option value="1000">All</option>
              </select>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Grouped Hotspot</th>
                  <th>Total Cycles</th>
                  <th>Avg Cycles / Invocation</th>
                  <th>Dispatches</th>
                  <th>Distinct Kernels</th>
                  <th>Distinct Ops</th>
                </tr>
              </thead>
              <tbody id="table-body"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="right-column">
        <div class="panel">
          <div class="panel-head">
            <h2>Selected Drilldown</h2>
            <p>Selection stays concise: hierarchy context, total-cycle emphasis, and correlation-quality cues for the chosen row.</p>
          </div>
          <div class="detail-body" id="detail-panel">
            <div class="placeholder">Select a row from the hotspot table to inspect its hierarchy chain, ranking weight, and kernel metadata.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const payload = {data_json};
    const rows = payload.rows.map((row, index) => ({{
      ...row,
      __index: index,
      gtpin_total_execution_cycles_num: parseInt(row.gtpin_total_execution_cycles || "0", 10) || 0,
      gtpin_invocation_count_num: parseInt(row.gtpin_invocation_count || "0", 10) || 0,
      gtpin_avg_execution_cycles_per_invocation_num: parseInt(row.gtpin_avg_execution_cycles_per_invocation || "0", 10) || 0,
      dispatch_count_num: parseInt(row.dispatch_count || "0", 10) || 0,
    }}));

    const views = payload.views;
    const maxRowCycles = Math.max(...rows.map(row => row.gtpin_total_execution_cycles_num), 0);
    const state = {{
      activeViewId: views[0].id,
      expanded: new Set(),
      selectedKey: "",
      rowLimit: 50,
    }};

    function formatNumber(value) {{
      return new Intl.NumberFormat("en-US").format(Number(value || 0));
    }}

    function formatPercent(value) {{
      return `${{(value * 100).toFixed(1)}}%`;
    }}

    function intensityColor(intensity) {{
      const hue = 208 - (intensity * 160);
      const sat = 84;
      const light = 58 - intensity * 12;
      return `hsl(${{hue}} ${{sat}}% ${{light}}%)`;
    }}

    function getActiveView() {{
      return views.find(view => view.id === state.activeViewId) || views[0];
    }}

    function makeGroupKey(parentKey, field, value) {{
      return `${{parentKey}}|${{field}}=${{value}}`;
    }}

    function buildGroupRows(sourceRows, groupKeys, depth, parentKey, lineage) {{
      if (depth >= groupKeys.length) {{
        return [];
      }}
      const field = groupKeys[depth];
      const grouped = new Map();
      for (const row of sourceRows) {{
        const value = String(row[field] || "(blank)");
        if (!grouped.has(value)) {{
          grouped.set(value, []);
        }}
        grouped.get(value).push(row);
      }}

      const items = [];
      for (const [value, childRows] of grouped.entries()) {{
        const totalCycles = childRows.reduce((sum, row) => sum + row.gtpin_total_execution_cycles_num, 0);
        const totalInvocations = childRows.reduce((sum, row) => sum + row.gtpin_invocation_count_num, 0);
        const dispatchCount = childRows.reduce((sum, row) => sum + row.dispatch_count_num, 0);
        const distinctKernels = new Set(childRows.map(row => row.kernel_entry)).size;
        const distinctOps = new Set(childRows.map(row => row.origin_op_name)).size;
        const distinctPrimitives = new Set(childRows.map(row => row.primitive_id)).size;
        const avgCycles = totalInvocations > 0 ? Math.floor(totalCycles / totalInvocations) : 0;
        const key = makeGroupKey(parentKey, field, value);
        const nextLineage = [...lineage, {{ field, value }}];
        const isLeaf = depth === groupKeys.length - 1;

        items.push({{
          kind: isLeaf ? "leaf" : "group",
          field,
          value,
          key,
          depth,
          lineage: nextLineage,
          totalCycles,
          totalInvocations,
          avgCycles,
          dispatchCount,
          distinctKernels,
          distinctOps,
          distinctPrimitives,
          childrenRows: childRows,
          leafRow: isLeaf && childRows.length === 1 ? childRows[0] : null,
          topRow: [...childRows].sort((a, b) => b.gtpin_total_execution_cycles_num - a.gtpin_total_execution_cycles_num)[0],
        }});
      }}

      items.sort((a, b) => b.totalCycles - a.totalCycles || a.value.localeCompare(b.value));
      return items;
    }}

    function flattenRows(sourceRows, groupKeys, parentKey = "root", depth = 0, lineage = []) {{
      const groupedRows = buildGroupRows(sourceRows, groupKeys, depth, parentKey, lineage);
      const flat = [];
      for (const item of groupedRows) {{
        flat.push(item);
        if (item.kind === "group" && state.expanded.has(item.key)) {{
          flat.push(...flattenRows(item.childrenRows, groupKeys, item.key, depth + 1, item.lineage));
        }}
      }}
      return flat;
    }}

    function buildHeaderText(item, viewId) {{
      if (item.kind === "leaf" && item.leafRow) {{
        const row = item.leafRow;
        return {{
          main: row.kernel_entry,
          sub: `${{row.origin_op_name}} | ${{row.implementation}}`,
        }};
      }}

      const topRow = item.topRow || {{}};
      const subParts = [];
      if (viewId !== "hierarchy" && topRow.component_path) {{
        subParts.push(topRow.component_path);
      }}
      if (topRow.origin_op_name && item.field !== "origin_op_name") {{
        subParts.push(topRow.origin_op_name);
      }}
      if (topRow.implementation && item.field !== "implementation" && viewId !== "implementation") {{
        subParts.push(topRow.implementation);
      }}
      if (item.field !== "kernel_entry") {{
        subParts.push(`${{item.distinctKernels}} kernels`);
      }}
      return {{
        main: item.value,
        sub: subParts.join(" | "),
      }};
    }}

    function renderTable() {{
      const view = getActiveView();
      const body = document.getElementById("table-body");
      const limitValue = state.rowLimit;
      const flattened = flattenRows(rows, view.group_keys).slice(0, limitValue);
      body.innerHTML = "";

      for (const item of flattened) {{
        const tr = document.createElement("tr");
        if (state.selectedKey === item.key) {{
          tr.classList.add("selected");
        }}
        const intensity = maxRowCycles > 0 ? item.totalCycles / maxRowCycles : 0;
        const header = buildHeaderText(item, view.id);
        const hasChildren = item.kind === "group";
        const indent = item.depth * 18;

        tr.innerHTML = `
          <td>
            <div class="row-label" style="padding-left:${{indent}}px">
              <button class="expand-btn ${{hasChildren ? "" : "disabled"}}" data-action="expand" data-key="${{item.key}}">${{hasChildren && state.expanded.has(item.key) ? "−" : "+"}}</button>
              <span class="intensity-dot" style="background:${{intensityColor(intensity)}}"></span>
              <div class="label-stack">
                <div class="label-main">${{escapeHtml(header.main)}}</div>
                <div class="label-sub">${{escapeHtml(header.sub || "Direct hotspot row")}}</div>
              </div>
            </div>
          </td>
          <td class="heat-cell">
            <div class="heat-metric">
              <span>${{formatNumber(item.totalCycles)}}</span>
              <div class="heat-bar"><div class="heat-fill" style="width:${{Math.max(2, intensity * 100)}}%"></div></div>
            </div>
            <div class="metric-note">${{formatPercent(intensity)}} of hottest row in unit</div>
          </td>
          <td>${{formatNumber(item.avgCycles)}}</td>
          <td>${{formatNumber(item.dispatchCount)}}</td>
          <td>${{formatNumber(item.distinctKernels)}}</td>
          <td>${{formatNumber(item.distinctOps)}}</td>
        `;

        tr.addEventListener("click", event => {{
          const button = event.target.closest("[data-action='expand']");
          if (button && hasChildren) {{
            event.stopPropagation();
            toggleExpand(item.key);
            return;
          }}
          state.selectedKey = item.key;
          renderTable();
          renderDetail(item, view);
        }});
        body.appendChild(tr);
      }}
    }}

    function escapeHtml(text) {{
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function toggleExpand(key) {{
      if (state.expanded.has(key)) {{
        state.expanded.delete(key);
      }} else {{
        state.expanded.add(key);
      }}
      renderTable();
    }}

    function renderTabs() {{
      const tabs = document.getElementById("view-tabs");
      tabs.innerHTML = "";
      for (const view of views) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = `view-tab${{view.id === state.activeViewId ? " active" : ""}}`;
        button.textContent = view.label;
        button.title = view.description;
        button.addEventListener("click", () => {{
          state.activeViewId = view.id;
          state.expanded.clear();
          state.selectedKey = "";
          renderTabs();
          renderTable();
          renderPlaceholder(view);
        }});
        tabs.appendChild(button);
      }}
    }}

    function renderPlaceholder(view) {{
      const panel = document.getElementById("detail-panel");
      panel.innerHTML = `<div class="placeholder">Select a row in the <strong>${{escapeHtml(view.label)}}</strong> view to inspect hierarchy context, cycle intensity, and correlation-quality metadata.</div>`;
    }}

    function buildCrumbs(item) {{
      const crumbs = [];
      for (const part of item.lineage) {{
        crumbs.push(`<span class="crumb">${{escapeHtml(part.value)}}</span>`);
      }}
      if (item.kind === "leaf" && item.leafRow) {{
        crumbs.push(`<span class="crumb">${{escapeHtml(item.leafRow.kernel_entry)}}</span>`);
      }}
      return crumbs.join("");
    }}

    function peerRanking(item) {{
      const sorted = [...item.childrenRows].sort((a, b) => b.gtpin_total_execution_cycles_num - a.gtpin_total_execution_cycles_num);
      const top = sorted[0] || null;
      const leaf = item.leafRow;
      if (!top || !leaf) {{
        return {{
          width: 0,
          message: "Grouped row. Expand further to inspect a single hotspot against its siblings.",
        }};
      }}
      const width = top.gtpin_total_execution_cycles_num > 0
        ? (leaf.gtpin_total_execution_cycles_num / top.gtpin_total_execution_cycles_num) * 100
        : 0;
      const rank = sorted.findIndex(row => row.__index === leaf.__index) + 1;
      return {{
        width,
        message: `Sibling rank ${{rank}} of ${{sorted.length}} in current branch`,
      }};
    }}

    function representativeKernels(item) {{
      const grouped = new Map();
      for (const row of item.childrenRows) {{
        const key = row.kernel_entry;
        if (!grouped.has(key)) {{
          grouped.set(key, {{
            kernel_entry: row.kernel_entry,
            totalCycles: 0,
            count: 0,
          }});
        }}
        const bucket = grouped.get(key);
        bucket.totalCycles += row.gtpin_total_execution_cycles_num;
        bucket.count += 1;
      }}
      return [...grouped.values()]
        .sort((a, b) => b.totalCycles - a.totalCycles || a.kernel_entry.localeCompare(b.kernel_entry))
        .slice(0, 4);
    }}

    function renderDetail(item, view) {{
      const panel = document.getElementById("detail-panel");
      const row = item.leafRow || item.topRow || null;
      const ranking = peerRanking(item);
      const intensity = maxRowCycles > 0 ? item.totalCycles / maxRowCycles : 0;
      const primitiveLine = row && row.primitive_id && row.primitive_id !== row.origin_op_name
        ? `<div><strong>Primitive:</strong> <span class="mono">${{escapeHtml(row.primitive_id)}}</span></div>`
        : "";
      const kernelLine = row && row.kernel_entry
        ? `<div><strong>Kernel:</strong> <span class="mono">${{escapeHtml(row.kernel_entry)}}</span></div>`
        : "";
      const dispatchLine = row && row.dispatch_ids
        ? `<div><strong>Dispatch IDs:</strong> <span class="mono">${{escapeHtml(row.dispatch_ids)}}</span></div>`
        : "";
      const repKernels = item.kind === "group"
        ? representativeKernels(item).map(kernel => `
            <div class="kernel-chip">
              <span class="mono">${{escapeHtml(kernel.kernel_entry)}}</span>
              <span class="kernel-chip-metric">${{formatNumber(kernel.totalCycles)}} cycles</span>
            </div>
          `).join("")
        : "";
      const panelSubtitle = item.kind === "leaf" && row
        ? `${{view.label}} drilldown | exact hotspot row`
        : `${{view.label}} drilldown | representative kernels for this grouped branch`;

      panel.innerHTML = `
        <div class="crumbs">${{buildCrumbs(item)}}</div>
        <div>
          <h3 class="detail-title">${{escapeHtml(item.kind === "leaf" && row ? row.kernel_entry : item.value)}}</h3>
          <p class="detail-sub">${{panelSubtitle}}${{row && item.kind === "leaf" ? ` | ${{row.origin_op_name}}` : ""}}</p>
        </div>
        <div class="detail-grid">
          <div class="detail-stat">
            <span class="detail-stat-label">Total Cycles</span>
            <div class="detail-stat-value">${{formatNumber(item.totalCycles)}}</div>
          </div>
          <div class="detail-stat">
            <span class="detail-stat-label">Avg Cycles / Invocation</span>
            <div class="detail-stat-value">${{formatNumber(item.avgCycles)}}</div>
          </div>
          <div class="detail-stat">
            <span class="detail-stat-label">Dispatches</span>
            <div class="detail-stat-value">${{formatNumber(item.dispatchCount)}}</div>
          </div>
          <div class="detail-stat">
            <span class="detail-stat-label">Distinct Kernels</span>
            <div class="detail-stat-value">${{formatNumber(item.distinctKernels)}}</div>
          </div>
        </div>
        <div class="peer-bar-shell">
          <div class="detail-sub">${{ranking.message}}</div>
          <div class="peer-bar"><span style="width:${{Math.max(2, ranking.width)}}%"></span></div>
          <div class="detail-sub">${{formatPercent(intensity)}} of hottest row in this inference</div>
        </div>
        <div class="quality-list">
          <div class="quality-item"><strong>Component Path</strong><span>${{escapeHtml(row?.component_path || item.value)}}</span></div>
          <div class="quality-item"><strong>Resolved Op Type</strong><span>${{escapeHtml(row?.resolved_op_type_name || "")}}</span></div>
          <div class="quality-item"><strong>Implementation</strong><span class="mono">${{escapeHtml(row?.implementation || "")}}</span></div>
          <div class="quality-item"><strong>Kernel Alignment</strong><span class="${{row?.kernel_alignment_status === "aligned" ? "quality-ok" : "quality-warn"}}">${{escapeHtml(row?.kernel_alignment_status || "grouped")}}</span></div>
          <div class="quality-item"><strong>Pointer Coverage</strong><span class="${{row?.pointer_coverage_match === "true" ? "quality-ok" : "quality-warn"}}">${{escapeHtml(row?.pointer_coverage_match || "grouped")}}</span></div>
          <div class="quality-item"><strong>OV Match</strong><span class="${{row?.ov_match === "matched" ? "quality-ok" : "quality-warn"}}">${{escapeHtml(row?.ov_match || "grouped")}}</span></div>
        </div>
        ${{item.kind === "group" ? `
          <div>
            <p class="detail-sub">Representative kernels in this branch</p>
            <div class="kernel-list">${{repKernels}}</div>
          </div>
        ` : ""}}
        <div class="small-list">
          <div><strong>Origin Op:</strong> <span class="mono">${{escapeHtml(row?.origin_op_name || "")}}</span></div>
          ${{primitiveLine}}
          ${{kernelLine}}
          ${{dispatchLine}}
          <div><strong>Execution Descriptor:</strong> <span class="mono">${{escapeHtml(row?.execution_descriptor || "")}}</span></div>
        </div>
      `;
    }}

    document.getElementById("row-limit").addEventListener("change", event => {{
      const value = parseInt(event.target.value, 10) || 50;
      state.rowLimit = value;
      renderTable();
    }});

    renderTabs();
    renderPlaceholder(getActiveView());
    renderTable();
  </script>
</body>
</html>
"""
    (output_dir / page_name).write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static hotspot-analysis HTML views from hotspot-first graph-prep bundles.")
    parser.add_argument("--graph-prep-root", required=True, help="Directory containing hotspot-first graph-prep bundle outputs")
    parser.add_argument("--output-dir", required=True, help="Directory for generated HTML files")
    args = parser.parse_args()

    graph_prep_root = Path(args.graph_prep_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = detect_bundle_prefix(graph_prep_root)
    hotspot_rows = read_csv(graph_prep_root / f"{prefix}_hotspot_table.csv")
    execution_units = read_csv(graph_prep_root / f"{prefix}_execution_units.csv")
    _summary = read_json(graph_prep_root / f"{prefix}_summary.json")

    hotspots_by_unit: Dict[str, List[Dict[str, str]]] = {}
    for row in hotspot_rows:
        unit_key = str(row.get("execution_unit_key", ""))
        hotspots_by_unit.setdefault(unit_key, []).append(row)

    unit_summaries = summarize_all_units(execution_units, hotspots_by_unit)
    render_index_page(
        output_dir=output_dir,
        title="GTPin Hotspot Analysis",
        prefix=prefix,
        unit_summaries=unit_summaries,
    )
    for summary in unit_summaries:
        render_unit_page(
            output_dir=output_dir,
            prefix=prefix,
            summary=summary,
            hotspot_rows=hotspots_by_unit.get(str(summary["execution_unit_key"]), []),
        )


if __name__ == "__main__":
    main()
