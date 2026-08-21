#!/usr/bin/env python3
"""
gsoc gtpin

Prepare hotspot-first correlation artifacts from:
1. dispatch_gtpin_kernel_metrics_join.csv
2. ov_topdown_primitive_rows*.csv

The goal is not to rebuild full topology. The goal is to expose:
- per-inference kernel hotspots with higher-level OpenVINO context
- a lightweight matched hierarchy for drilldown
- a compact graph structure that can power later visualization
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HOTSPOT_FIELDS = [
    "execution_unit_key",
    "logical_execution_index",
    "net_id",
    "iteration",
    "is_internal_network_candidate",
    "component_path",
    "origin_op_name",
    "origin_op_type_name",
    "resolved_op_type_name",
    "primitive_id",
    "original_primitive_id",
    "primitive_type",
    "implementation",
    "exec_id",
    "kernel_entry",
    "gtpin_kernel",
    "dispatch_count",
    "dispatch_ids",
    "ov_match",
    "kernel_alignment_status",
    "kernel_entry_alignment_verified",
    "pointer_coverage_match",
    "ov_kernel_entry_present",
    "gtpin_source_kind",
    "gtpin_invocation_count",
    "gtpin_total_execution_cycles",
    "gtpin_avg_execution_cycles_per_invocation",
    "execution_descriptor",
]

HIERARCHY_FIELDS = [
    "execution_unit_key",
    "logical_execution_index",
    "net_id",
    "iteration",
    "component_path",
    "origin_op_name",
    "origin_op_type_name",
    "resolved_op_type_name",
    "primitive_id",
    "original_primitive_id",
    "primitive_type",
    "implementation",
    "exec_id",
    "is_input",
    "is_output",
    "dispatch_count",
    "kernel_entry_count",
    "kernel_entries",
    "gtpin_invocation_count",
    "gtpin_total_execution_cycles",
]

GRAPH_NODE_FIELDS = [
    "execution_unit_key",
    "logical_execution_index",
    "net_id",
    "iteration",
    "node_id",
    "parent_node_id",
    "node_type",
    "display_name",
    "component_path",
    "origin_op_name",
    "resolved_op_type_name",
    "primitive_id",
    "primitive_type",
    "implementation",
    "kernel_entry",
    "dispatch_count",
    "gtpin_invocation_count",
    "gtpin_total_execution_cycles",
]

GRAPH_EDGE_FIELDS = [
    "execution_unit_key",
    "logical_execution_index",
    "edge_type",
    "source_node_id",
    "target_node_id",
]


def to_int(value: str, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def split_semicolon(value: str) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def derive_component_path(origin_op_name: str) -> str:
    text = str(origin_op_name).strip().strip("/")
    if not text:
        return "(root)"

    parts = [part for part in text.split("/") if part]
    if parts and parts[0] == "model":
        parts = parts[1:]

    if len(parts) <= 1:
        return "(root)"

    head = parts[0]
    if head.startswith("layers."):
        if len(parts) >= 2 and parts[1] in {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}:
            return f"layers.*/{parts[1]}"
        return "layers.*"

    if head in {"embed_tokens", "rotary_emb", "lm_head"}:
        return head

    return "(root)"


def derive_resolved_op_type(origin_op_type_name: str, primitive_type: str, primitive_id: str) -> str:
    explicit = str(origin_op_type_name).strip()
    if explicit:
        return explicit

    primitive_kind = str(primitive_type).strip()
    if primitive_kind:
        return primitive_kind

    primitive_prefix = str(primitive_id).split(":", 1)[0].strip()
    return primitive_prefix or "Unknown"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def join_unique_strings(values: Iterable[object]) -> str:
    items = sorted({str(value).strip() for value in values if str(value).strip()})
    return ";".join(items)


def normalize_dispatch_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["net_id_num"] = to_int(row.get("net_id_num", row.get("net_id", "")), default=-1)
        item["iteration_num"] = to_int(row.get("iteration_num", row.get("iteration", "")), default=-1)
        item["dispatch_id_num"] = to_int(row.get("dispatch_id", ""), default=-1)
        item["dispatch_index_num"] = to_int(row.get("dispatch_index", ""), default=-1)
        item["gtpin_total_execution_cycles_num"] = to_int(row.get("gtpin_total_execution_cycles", ""), default=0)
        item["gtpin_invocation_count_num"] = to_int(row.get("gtpin_invocation_count", ""), default=0)
        item["ov_kernel_entry_present_bool"] = to_bool(row.get("ov_kernel_entry_present", ""))
        item["pointer_coverage_match_bool"] = to_bool(row.get("pointer_coverage_match", ""))
        if not item.get("execution_unit_key"):
            item["execution_unit_key"] = f"n{item['net_id_num']}_i{item['iteration_num']}"
        normalized.append(item)
    return normalized


def normalize_topdown_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    has_iteration = bool(rows) and "iteration" in rows[0]
    has_execution_unit = bool(rows) and "execution_unit_key" in rows[0]
    has_origin_type = bool(rows) and "origin_op_type_name" in rows[0]

    for row in rows:
        item = dict(row)
        item["net_id_num"] = to_int(row.get("net_id", ""), default=-1)
        item["iteration_num"] = to_int(row.get("iteration", ""), default=-1 if has_iteration else -1)
        item["exec_id_num"] = to_int(row.get("exec_id", ""), default=-1)
        item["is_input_bool"] = to_bool(row.get("is_input", ""))
        item["is_output_bool"] = to_bool(row.get("is_output", ""))
        item["dependencies_list"] = split_semicolon(row.get("dependencies", ""))
        item["users_list"] = split_semicolon(row.get("users", ""))
        item["fused_ids_list"] = split_semicolon(row.get("fused_ids", ""))
        item["origin_op_type_name"] = row.get("origin_op_type_name", "") if has_origin_type else ""
        item["resolved_op_type_name"] = derive_resolved_op_type(
            item["origin_op_type_name"],
            row.get("primitive_type", ""),
            row.get("primitive_id", ""),
        )
        item["component_path"] = derive_component_path(row.get("origin_op_name", ""))
        item["execution_unit_key"] = row.get("execution_unit_key", "") if has_execution_unit else ""
        item["topdown_mode"] = "inference_scoped" if has_iteration and has_execution_unit else "network_scoped"
        normalized.append(item)
    return normalized


def build_execution_units(dispatch_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    units: Dict[str, Dict[str, object]] = {}
    for row in dispatch_rows:
        key = str(row["execution_unit_key"])
        if key not in units:
            units[key] = {
                "execution_unit_key": key,
                "net_id": row.get("net_id", ""),
                "net_id_num": row["net_id_num"],
                "iteration": row.get("iteration", ""),
                "iteration_num": row["iteration_num"],
                "is_internal_network_candidate": row.get("is_internal_network_candidate", ""),
                "logical_execution_index": "",
                "dispatch_rows_total": 0,
                "matched_dispatch_rows": 0,
                "kernel_aligned_rows": 0,
            }
        unit = units[key]
        unit["dispatch_rows_total"] += 1
        if row.get("ov_match") == "matched":
            unit["matched_dispatch_rows"] += 1
        if row.get("kernel_alignment_status") == "aligned":
            unit["kernel_aligned_rows"] += 1

    steady_units = sorted(
        [unit for unit in units.values() if int(unit["net_id_num"]) != 0],
        key=lambda item: (int(item["net_id_num"]), int(item["iteration_num"]), str(item["execution_unit_key"])),
    )
    for logical_index, unit in enumerate(steady_units):
        unit["logical_execution_index"] = logical_index
    return units


def expand_topdown_by_execution_unit(
    topdown_rows: List[Dict[str, object]],
    execution_units: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    by_net: Dict[int, List[str]] = defaultdict(list)
    for unit_key, unit in execution_units.items():
        by_net[int(unit["net_id_num"])].append(unit_key)

    expanded: List[Dict[str, object]] = []
    topology_mode_by_unit: Dict[str, str] = {}

    for row in topdown_rows:
        if row["topdown_mode"] == "inference_scoped":
            unit_key = str(row["execution_unit_key"])
            if not unit_key:
                unit_key = f"n{row['net_id_num']}_i{row['iteration_num']}"
                row["execution_unit_key"] = unit_key
            topology_mode_by_unit[unit_key] = "inference_scoped"
            expanded.append(row)
            continue

        candidate_units = sorted(by_net.get(int(row["net_id_num"]), []))
        if not candidate_units:
            replicated = dict(row)
            replicated["execution_unit_key"] = ""
            replicated["iteration_num"] = -1
            replicated["topology_scope"] = "network_without_dispatch_match"
            expanded.append(replicated)
            continue

        for unit_key in candidate_units:
            unit = execution_units[unit_key]
            replicated = dict(row)
            replicated["execution_unit_key"] = unit_key
            replicated["iteration_num"] = unit["iteration_num"]
            replicated["iteration"] = unit["iteration"]
            replicated["topology_scope"] = "replicated_from_network_topology"
            expanded.append(replicated)
            topology_mode_by_unit.setdefault(unit_key, "replicated_from_network_topology")

    return expanded, topology_mode_by_unit


def get_logical_execution_index(execution_units: Dict[str, Dict[str, object]], execution_unit_key: str) -> str:
    unit = execution_units.get(execution_unit_key)
    if not unit:
        return ""
    return str(unit.get("logical_execution_index", ""))


def build_matched_hierarchy_rows(
    topdown_rows: List[Dict[str, object]],
    dispatch_rows: List[Dict[str, object]],
    execution_units: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    dispatch_by_unit_primitive: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in dispatch_rows:
        dispatch_by_unit_primitive[(str(row["execution_unit_key"]), str(row.get("primitive_id", "")))].append(row)

    matched_rows: List[Dict[str, object]] = []
    for row in topdown_rows:
        unit_key = str(row.get("execution_unit_key", ""))
        primitive_id = str(row.get("primitive_id", ""))
        matching_dispatch_rows = dispatch_by_unit_primitive.get((unit_key, primitive_id), [])
        if not matching_dispatch_rows:
            continue

        kernel_entries = sorted({
            str(dispatch_row.get("kernel_entry", ""))
            for dispatch_row in matching_dispatch_rows
            if str(dispatch_row.get("kernel_entry", "")).strip()
        })
        matched_rows.append({
            "execution_unit_key": unit_key,
            "logical_execution_index": get_logical_execution_index(execution_units, unit_key),
            "net_id": row.get("net_id", ""),
            "iteration": row.get("iteration", ""),
            "component_path": row.get("component_path", "(root)"),
            "origin_op_name": row.get("origin_op_name", ""),
            "origin_op_type_name": row.get("origin_op_type_name", ""),
            "resolved_op_type_name": row.get("resolved_op_type_name", "Unknown"),
            "primitive_id": primitive_id,
            "original_primitive_id": row.get("original_primitive_id", ""),
            "primitive_type": row.get("primitive_type", ""),
            "implementation": row.get("implementation", ""),
            "exec_id": row.get("exec_id", ""),
            "is_input": row.get("is_input", ""),
            "is_output": row.get("is_output", ""),
            "dispatch_count": len(matching_dispatch_rows),
            "kernel_entry_count": len(kernel_entries),
            "kernel_entries": ";".join(kernel_entries),
            "gtpin_invocation_count": sum(int(item["gtpin_invocation_count_num"]) for item in matching_dispatch_rows),
            "gtpin_total_execution_cycles": sum(int(item["gtpin_total_execution_cycles_num"]) for item in matching_dispatch_rows),
        })
    return matched_rows


def build_hotspot_rows(
    matched_hierarchy_rows: List[Dict[str, object]],
    dispatch_rows: List[Dict[str, object]],
    execution_units: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    hierarchy_by_unit_primitive: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in matched_hierarchy_rows:
        hierarchy_by_unit_primitive[(str(row["execution_unit_key"]), str(row["primitive_id"]))] = row

    grouped_dispatch: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in dispatch_rows:
        if row.get("ov_match") != "matched":
            continue
        unit_key = str(row.get("execution_unit_key", ""))
        primitive_id = str(row.get("primitive_id", ""))
        hierarchy_row = hierarchy_by_unit_primitive.get((unit_key, primitive_id))
        if not hierarchy_row:
            continue
        kernel_entry = str(row.get("kernel_entry", ""))
        grouped_dispatch[(unit_key, primitive_id, kernel_entry)].append(row)

    hotspot_rows: List[Dict[str, object]] = []
    for key, rows in sorted(grouped_dispatch.items()):
        unit_key, primitive_id, kernel_entry = key
        hierarchy_row = hierarchy_by_unit_primitive[(unit_key, primitive_id)]
        first = rows[0]
        invocation_total = sum(int(row["gtpin_invocation_count_num"]) for row in rows)
        cycles_total = sum(int(row["gtpin_total_execution_cycles_num"]) for row in rows)
        avg_cycles = int(cycles_total / invocation_total) if invocation_total else 0

        hotspot_rows.append({
            "execution_unit_key": unit_key,
            "logical_execution_index": get_logical_execution_index(execution_units, unit_key),
            "net_id": first.get("net_id", ""),
            "iteration": first.get("iteration", ""),
            "is_internal_network_candidate": first.get("is_internal_network_candidate", ""),
            "component_path": hierarchy_row.get("component_path", "(root)"),
            "origin_op_name": hierarchy_row.get("origin_op_name", ""),
            "origin_op_type_name": hierarchy_row.get("origin_op_type_name", ""),
            "resolved_op_type_name": hierarchy_row.get("resolved_op_type_name", "Unknown"),
            "primitive_id": primitive_id,
            "original_primitive_id": hierarchy_row.get("original_primitive_id", ""),
            "primitive_type": first.get("primitive_type", ""),
            "implementation": first.get("implementation", ""),
            "exec_id": hierarchy_row.get("exec_id", ""),
            "kernel_entry": kernel_entry,
            "gtpin_kernel": join_unique_strings(row.get("gtpin_kernel", "") for row in rows),
            "dispatch_count": len(rows),
            "dispatch_ids": join_unique_strings(row.get("dispatch_id", "") for row in rows),
            "ov_match": join_unique_strings(row.get("ov_match", "") for row in rows),
            "kernel_alignment_status": join_unique_strings(row.get("kernel_alignment_status", "") for row in rows),
            "kernel_entry_alignment_verified": join_unique_strings(row.get("kernel_entry_alignment_verified", "") for row in rows),
            "pointer_coverage_match": join_unique_strings(row.get("pointer_coverage_match", "") for row in rows),
            "ov_kernel_entry_present": join_unique_strings(row.get("ov_kernel_entry_present", "") for row in rows),
            "gtpin_source_kind": join_unique_strings(row.get("gtpin_source_kind", "") for row in rows),
            "gtpin_invocation_count": invocation_total,
            "gtpin_total_execution_cycles": cycles_total,
            "gtpin_avg_execution_cycles_per_invocation": avg_cycles,
            "execution_descriptor": join_unique_strings(row.get("execution_descriptor", "") for row in rows),
        })

    hotspot_rows.sort(
        key=lambda row: (
            str(row["execution_unit_key"]),
            -to_int(row["gtpin_total_execution_cycles"], default=0),
            str(row["component_path"]),
            str(row["origin_op_name"]),
            str(row["primitive_id"]),
            str(row["kernel_entry"]),
        )
    )
    return hotspot_rows


def build_graph_structure_rows(
    hotspot_rows: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    component_nodes: Dict[Tuple[str, str], Dict[str, object]] = {}
    op_type_nodes: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    op_nodes: Dict[Tuple[str, str], Dict[str, object]] = {}
    primitive_nodes: Dict[Tuple[str, str], Dict[str, object]] = {}
    kernel_nodes: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    edges: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    def merge_metric(node: Dict[str, object], row: Dict[str, object]) -> None:
        node["dispatch_count"] = int(node.get("dispatch_count", 0)) + to_int(row.get("dispatch_count", 0), default=0)
        node["gtpin_invocation_count"] = int(node.get("gtpin_invocation_count", 0)) + to_int(row.get("gtpin_invocation_count", 0), default=0)
        node["gtpin_total_execution_cycles"] = int(node.get("gtpin_total_execution_cycles", 0)) + to_int(row.get("gtpin_total_execution_cycles", 0), default=0)

    for row in hotspot_rows:
        unit_key = str(row["execution_unit_key"])
        logical_index = str(row.get("logical_execution_index", ""))
        net_id = str(row.get("net_id", ""))
        iteration = str(row.get("iteration", ""))
        component_path = str(row.get("component_path", "(root)"))
        resolved_op_type_name = str(row.get("resolved_op_type_name", "Unknown"))
        origin_op_name = str(row.get("origin_op_name", ""))
        primitive_id = str(row.get("primitive_id", ""))
        kernel_entry = str(row.get("kernel_entry", ""))

        component_key = (unit_key, component_path)
        if component_key not in component_nodes:
            component_nodes[component_key] = {
                "execution_unit_key": unit_key,
                "logical_execution_index": logical_index,
                "net_id": net_id,
                "iteration": iteration,
                "node_id": f"{unit_key}::component_group::{component_path}",
                "parent_node_id": "",
                "node_type": "component_group",
                "display_name": component_path,
                "component_path": component_path,
                "origin_op_name": "",
                "resolved_op_type_name": "",
                "primitive_id": "",
                "primitive_type": "",
                "implementation": "",
                "kernel_entry": "",
                "dispatch_count": 0,
                "gtpin_invocation_count": 0,
                "gtpin_total_execution_cycles": 0,
            }
        merge_metric(component_nodes[component_key], row)

        op_type_key = (unit_key, component_path, resolved_op_type_name)
        if op_type_key not in op_type_nodes:
            op_type_nodes[op_type_key] = {
                "execution_unit_key": unit_key,
                "logical_execution_index": logical_index,
                "net_id": net_id,
                "iteration": iteration,
                "node_id": f"{unit_key}::op_type::{component_path}::{resolved_op_type_name}",
                "parent_node_id": component_nodes[component_key]["node_id"],
                "node_type": "op_type",
                "display_name": resolved_op_type_name,
                "component_path": component_path,
                "origin_op_name": "",
                "resolved_op_type_name": resolved_op_type_name,
                "primitive_id": "",
                "primitive_type": "",
                "implementation": "",
                "kernel_entry": "",
                "dispatch_count": 0,
                "gtpin_invocation_count": 0,
                "gtpin_total_execution_cycles": 0,
            }
        merge_metric(op_type_nodes[op_type_key], row)

        op_key = (unit_key, origin_op_name)
        if op_key not in op_nodes:
            op_nodes[op_key] = {
                "execution_unit_key": unit_key,
                "logical_execution_index": logical_index,
                "net_id": net_id,
                "iteration": iteration,
                "node_id": f"{unit_key}::op::{origin_op_name}",
                "parent_node_id": op_type_nodes[op_type_key]["node_id"],
                "node_type": "op",
                "display_name": origin_op_name,
                "component_path": component_path,
                "origin_op_name": origin_op_name,
                "resolved_op_type_name": resolved_op_type_name,
                "primitive_id": "",
                "primitive_type": "",
                "implementation": "",
                "kernel_entry": "",
                "dispatch_count": 0,
                "gtpin_invocation_count": 0,
                "gtpin_total_execution_cycles": 0,
            }
        merge_metric(op_nodes[op_key], row)

        primitive_key = (unit_key, primitive_id)
        if primitive_key not in primitive_nodes:
            primitive_nodes[primitive_key] = {
                "execution_unit_key": unit_key,
                "logical_execution_index": logical_index,
                "net_id": net_id,
                "iteration": iteration,
                "node_id": f"{unit_key}::primitive::{primitive_id}",
                "parent_node_id": op_nodes[op_key]["node_id"],
                "node_type": "primitive",
                "display_name": primitive_id,
                "component_path": component_path,
                "origin_op_name": origin_op_name,
                "resolved_op_type_name": resolved_op_type_name,
                "primitive_id": primitive_id,
                "primitive_type": str(row.get("primitive_type", "")),
                "implementation": str(row.get("implementation", "")),
                "kernel_entry": "",
                "dispatch_count": 0,
                "gtpin_invocation_count": 0,
                "gtpin_total_execution_cycles": 0,
            }
        merge_metric(primitive_nodes[primitive_key], row)

        kernel_key = (unit_key, primitive_id, kernel_entry)
        if kernel_key not in kernel_nodes:
            kernel_nodes[kernel_key] = {
                "execution_unit_key": unit_key,
                "logical_execution_index": logical_index,
                "net_id": net_id,
                "iteration": iteration,
                "node_id": f"{unit_key}::kernel::{primitive_id}::{kernel_entry}",
                "parent_node_id": primitive_nodes[primitive_key]["node_id"],
                "node_type": "kernel",
                "display_name": kernel_entry,
                "component_path": component_path,
                "origin_op_name": origin_op_name,
                "resolved_op_type_name": resolved_op_type_name,
                "primitive_id": primitive_id,
                "primitive_type": str(row.get("primitive_type", "")),
                "implementation": str(row.get("implementation", "")),
                "kernel_entry": kernel_entry,
                "dispatch_count": 0,
                "gtpin_invocation_count": 0,
                "gtpin_total_execution_cycles": 0,
            }
        merge_metric(kernel_nodes[kernel_key], row)

        edge_specs = [
            ("component_group_to_op_type", component_nodes[component_key]["node_id"], op_type_nodes[op_type_key]["node_id"]),
            ("op_type_to_op", op_type_nodes[op_type_key]["node_id"], op_nodes[op_key]["node_id"]),
            ("op_to_primitive", op_nodes[op_key]["node_id"], primitive_nodes[primitive_key]["node_id"]),
            ("primitive_to_kernel", primitive_nodes[primitive_key]["node_id"], kernel_nodes[kernel_key]["node_id"]),
        ]
        for edge_type, source_id, target_id in edge_specs:
            edge_key = (unit_key, edge_type, source_id, target_id)
            if edge_key not in edges:
                edges[edge_key] = {
                    "execution_unit_key": unit_key,
                    "logical_execution_index": logical_index,
                    "edge_type": edge_type,
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                }

    graph_nodes = (
        list(component_nodes.values())
        + list(op_type_nodes.values())
        + list(op_nodes.values())
        + list(primitive_nodes.values())
        + list(kernel_nodes.values())
    )
    graph_nodes.sort(key=lambda row: (str(row["execution_unit_key"]), str(row["node_type"]), str(row["node_id"])))
    graph_edges = list(edges.values())
    graph_edges.sort(key=lambda row: (str(row["execution_unit_key"]), str(row["edge_type"]), str(row["source_node_id"]), str(row["target_node_id"])))
    return graph_nodes, graph_edges


def aggregate_graph_rows(
    dispatch_rows: List[Dict[str, object]],
    topdown_rows: List[Dict[str, object]],
    execution_units: Dict[str, Dict[str, object]],
    topology_mode_by_unit: Dict[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    dispatch_by_unit_primitive: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in dispatch_rows:
        dispatch_by_unit_primitive[(str(row["execution_unit_key"]), str(row.get("primitive_id", "")))].append(row)

    component_group_nodes: List[Dict[str, object]] = []
    op_type_nodes: List[Dict[str, object]] = []
    layer_nodes: List[Dict[str, object]] = []
    primitive_nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []

    primitive_node_ids: Dict[Tuple[str, str], str] = {}
    layer_node_ids: Dict[Tuple[str, str], str] = {}
    component_group_node_ids: Dict[Tuple[str, str], str] = {}
    op_type_node_ids: Dict[Tuple[str, str, str], str] = {}
    unmapped_topdown_rows = 0
    multi_kernel_primitives = 0

    grouped_topdown: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in topdown_rows:
        grouped_topdown[(str(row.get("execution_unit_key", "")), str(row.get("primitive_id", "")))].append(row)

    grouped_component_paths: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    grouped_op_types: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    grouped_layers: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in topdown_rows:
        grouped_component_paths[(str(row.get("execution_unit_key", "")), str(row.get("component_path", "(root)")))].append(row)
        grouped_op_types[(
            str(row.get("execution_unit_key", "")),
            str(row.get("component_path", "(root)")),
            str(row.get("resolved_op_type_name", "Unknown")),
        )].append(row)
        grouped_layers[(str(row.get("execution_unit_key", "")), str(row.get("origin_op_name", "")))].append(row)

    for (unit_key, component_path), rows in sorted(grouped_component_paths.items()):
        first = rows[0]
        total_cycles = 0
        dispatch_children = 0
        primitive_ids = {str(row.get("primitive_id", "")) for row in rows}
        layer_names = {str(row.get("origin_op_name", "")) for row in rows}
        for row in rows:
            primitive_dispatch_rows = dispatch_by_unit_primitive.get((unit_key, str(row.get("primitive_id", ""))), [])
            dispatch_children += len(primitive_dispatch_rows)
            total_cycles += sum(int(item["gtpin_total_execution_cycles_num"]) for item in primitive_dispatch_rows)

        node_id = f"{unit_key}::component_group::{component_path}"
        component_group_node_ids[(unit_key, component_path)] = node_id
        component_group_nodes.append({
            "node_id": node_id,
            "parent_node_id": "",
            "execution_unit_key": unit_key,
            "net_id": first.get("net_id", ""),
            "iteration": first.get("iteration", ""),
            "node_type": "component_group",
            "label": component_path,
            "component_path": component_path,
            "origin_op_name": "",
            "origin_op_type_name": "",
            "resolved_op_type_name": "",
            "primitive_id": "",
            "original_primitive_id": "",
            "primitive_type": "",
            "implementation": "",
            "exec_id": "",
            "dispatch_rows": dispatch_children,
            "kernel_entry_count": 0,
            "kernel_entries": "",
            "total_cycles": total_cycles,
            "dependencies": "",
            "users": "",
            "fused_ids": "",
            "topology_mode": topology_mode_by_unit.get(unit_key, "unknown"),
            "mapping_status": "aggregated",
            "layer_children": len(layer_names),
            "primitive_children": len(primitive_ids),
        })

    for (unit_key, component_path, resolved_op_type_name), rows in sorted(grouped_op_types.items()):
        first = rows[0]
        total_cycles = 0
        dispatch_children = 0
        primitive_ids = {str(row.get("primitive_id", "")) for row in rows}
        layer_names = {str(row.get("origin_op_name", "")) for row in rows}
        for row in rows:
            primitive_dispatch_rows = dispatch_by_unit_primitive.get((unit_key, str(row.get("primitive_id", ""))), [])
            dispatch_children += len(primitive_dispatch_rows)
            total_cycles += sum(int(item["gtpin_total_execution_cycles_num"]) for item in primitive_dispatch_rows)

        node_id = f"{unit_key}::op_type::{component_path}::{resolved_op_type_name}"
        op_type_node_ids[(unit_key, component_path, resolved_op_type_name)] = node_id
        parent_component_id = component_group_node_ids.get((unit_key, component_path), "")
        op_type_nodes.append({
            "node_id": node_id,
            "parent_node_id": parent_component_id,
            "execution_unit_key": unit_key,
            "net_id": first.get("net_id", ""),
            "iteration": first.get("iteration", ""),
            "node_type": "op_type",
            "label": resolved_op_type_name,
            "component_path": component_path,
            "origin_op_name": "",
            "origin_op_type_name": first.get("origin_op_type_name", ""),
            "resolved_op_type_name": resolved_op_type_name,
            "primitive_id": "",
            "original_primitive_id": "",
            "primitive_type": "",
            "implementation": "",
            "exec_id": "",
            "dispatch_rows": dispatch_children,
            "kernel_entry_count": 0,
            "kernel_entries": "",
            "total_cycles": total_cycles,
            "dependencies": "",
            "users": "",
            "fused_ids": "",
            "topology_mode": topology_mode_by_unit.get(unit_key, "unknown"),
            "mapping_status": "aggregated",
            "layer_children": len(layer_names),
            "primitive_children": len(primitive_ids),
        })
        if parent_component_id:
            edges.append({
                "execution_unit_key": unit_key,
                "edge_type": "component_group_to_op_type",
                "source_node_id": parent_component_id,
                "target_node_id": node_id,
                "source_label": component_path,
                "target_label": resolved_op_type_name,
                "semantic": "hierarchy",
            })

    for (unit_key, origin_op_name), rows in sorted(grouped_layers.items()):
        first = rows[0]
        dispatch_children = 0
        total_cycles = 0
        for row in rows:
            primitive_dispatch_rows = dispatch_by_unit_primitive.get((unit_key, str(row.get("primitive_id", ""))), [])
            if primitive_dispatch_rows:
                dispatch_children += len(primitive_dispatch_rows)
                total_cycles += sum(int(item["gtpin_total_execution_cycles_num"]) for item in primitive_dispatch_rows)

        node_id = f"{unit_key}::layer::{origin_op_name}"
        layer_node_ids[(unit_key, origin_op_name)] = node_id
        parent_op_type_id = op_type_node_ids.get((
            unit_key,
            str(first.get("component_path", "(root)")),
            str(first.get("resolved_op_type_name", "Unknown")),
        ), "")
        layer_nodes.append({
            "node_id": node_id,
            "parent_node_id": parent_op_type_id,
            "execution_unit_key": unit_key,
            "net_id": first.get("net_id", ""),
            "iteration": first.get("iteration", ""),
            "node_type": "origin_op",
            "label": origin_op_name,
            "component_path": first.get("component_path", "(root)"),
            "origin_op_name": origin_op_name,
            "origin_op_type_name": first.get("origin_op_type_name", ""),
            "resolved_op_type_name": first.get("resolved_op_type_name", "Unknown"),
            "primitive_children": len({str(row.get("primitive_id", "")) for row in rows}),
            "dispatch_rows": dispatch_children,
            "total_cycles": total_cycles,
            "topology_mode": topology_mode_by_unit.get(unit_key, "unknown"),
            "mapping_status": "aggregated",
            "layer_children": 0,
        })
        if parent_op_type_id:
            edges.append({
                "execution_unit_key": unit_key,
                "edge_type": "op_type_to_layer",
                "source_node_id": parent_op_type_id,
                "target_node_id": node_id,
                "source_label": first.get("resolved_op_type_name", "Unknown"),
                "target_label": origin_op_name,
                "semantic": "hierarchy",
            })

    for (unit_key, primitive_id), rows in sorted(grouped_topdown.items()):
        first = rows[0]
        matching_dispatch_rows = dispatch_by_unit_primitive.get((unit_key, primitive_id), [])
        kernel_entries = sorted({
            str(row.get("kernel_entry", ""))
            for row in matching_dispatch_rows
            if str(row.get("kernel_entry", "")).strip()
        })
        total_cycles = sum(int(row["gtpin_total_execution_cycles_num"]) for row in matching_dispatch_rows)
        dispatch_count = len(matching_dispatch_rows)
        if not matching_dispatch_rows:
            unmapped_topdown_rows += 1
        if len(kernel_entries) > 1:
            multi_kernel_primitives += 1

        primitive_node_id = f"{unit_key}::primitive::{primitive_id}"
        primitive_node_ids[(unit_key, primitive_id)] = primitive_node_id
        parent_layer_id = layer_node_ids.get((unit_key, str(first.get("origin_op_name", ""))), "")

        primitive_nodes.append({
            "node_id": primitive_node_id,
            "parent_node_id": parent_layer_id,
            "execution_unit_key": unit_key,
            "net_id": first.get("net_id", ""),
            "iteration": first.get("iteration", ""),
            "node_type": "primitive",
            "label": primitive_id,
            "component_path": first.get("component_path", "(root)"),
            "origin_op_name": first.get("origin_op_name", ""),
            "origin_op_type_name": first.get("origin_op_type_name", ""),
            "resolved_op_type_name": first.get("resolved_op_type_name", "Unknown"),
            "primitive_id": primitive_id,
            "original_primitive_id": first.get("original_primitive_id", ""),
            "primitive_type": first.get("primitive_type", ""),
            "implementation": first.get("implementation", ""),
            "exec_id": first.get("exec_id", ""),
            "dispatch_rows": dispatch_count,
            "kernel_entry_count": len(kernel_entries),
            "kernel_entries": ";".join(kernel_entries),
            "total_cycles": total_cycles,
            "dependencies": first.get("dependencies", ""),
            "users": first.get("users", ""),
            "fused_ids": first.get("fused_ids", ""),
            "topology_mode": topology_mode_by_unit.get(unit_key, "unknown"),
            "mapping_status": "mapped" if matching_dispatch_rows else "topology_only",
            "layer_children": 0,
            "primitive_children": 0,
        })

        if parent_layer_id:
            edges.append({
                "execution_unit_key": unit_key,
                "edge_type": "layer_to_primitive",
                "source_node_id": parent_layer_id,
                "target_node_id": primitive_node_id,
                "source_label": first.get("origin_op_name", ""),
                "target_label": primitive_id,
                "semantic": "hierarchy",
            })

        for kernel_entry in kernel_entries:
            kernel_rows = [row for row in matching_dispatch_rows if str(row.get("kernel_entry", "")) == kernel_entry]
            kernel_node_id = f"{unit_key}::kernel::{primitive_id}::{kernel_entry}"
            primitive_nodes.append({
                "node_id": kernel_node_id,
                "parent_node_id": primitive_node_id,
                "execution_unit_key": unit_key,
                "net_id": first.get("net_id", ""),
                "iteration": first.get("iteration", ""),
                "node_type": "kernel",
                "label": kernel_entry,
                "component_path": first.get("component_path", "(root)"),
                "origin_op_name": first.get("origin_op_name", ""),
                "origin_op_type_name": first.get("origin_op_type_name", ""),
                "resolved_op_type_name": first.get("resolved_op_type_name", "Unknown"),
                "primitive_id": primitive_id,
                "original_primitive_id": first.get("original_primitive_id", ""),
                "primitive_type": first.get("primitive_type", ""),
                "implementation": first.get("implementation", ""),
                "exec_id": first.get("exec_id", ""),
                "dispatch_rows": len(kernel_rows),
                "kernel_entry_count": 1,
                "kernel_entries": kernel_entry,
                "total_cycles": sum(int(row["gtpin_total_execution_cycles_num"]) for row in kernel_rows),
                "dependencies": "",
                "users": "",
                "fused_ids": "",
                "topology_mode": topology_mode_by_unit.get(unit_key, "unknown"),
                "mapping_status": "mapped_kernel",
                "layer_children": 0,
                "primitive_children": 0,
            })
            edges.append({
                "execution_unit_key": unit_key,
                "edge_type": "primitive_to_kernel",
                "source_node_id": primitive_node_id,
                "target_node_id": kernel_node_id,
                "source_label": primitive_id,
                "target_label": kernel_entry,
                "semantic": "hierarchy",
            })

    for row in topdown_rows:
        unit_key = str(row.get("execution_unit_key", ""))
        primitive_id = str(row.get("primitive_id", ""))
        source_node_id = primitive_node_ids.get((unit_key, primitive_id))
        if not source_node_id:
            continue
        for dep in row.get("dependencies_list", []):
            target_node_id = primitive_node_ids.get((unit_key, dep))
            edges.append({
                "execution_unit_key": unit_key,
                "edge_type": "primitive_dependency",
                "source_node_id": target_node_id or f"{unit_key}::primitive::{dep}",
                "target_node_id": source_node_id,
                "source_label": dep,
                "target_label": primitive_id,
                "semantic": "dependency",
            })

    dispatch_without_topdown = 0
    for row in dispatch_rows:
        key = (str(row["execution_unit_key"]), str(row.get("primitive_id", "")))
        if key not in grouped_topdown:
            dispatch_without_topdown += 1

    execution_unit_rows: List[Dict[str, object]] = []
    for unit_key, unit in sorted(execution_units.items()):
        unit_topdown = [row for row in topdown_rows if str(row.get("execution_unit_key", "")) == unit_key]
        unit_mapped = [row for row in unit_topdown if dispatch_by_unit_primitive.get((unit_key, str(row.get("primitive_id", ""))), [])]
        execution_unit_rows.append({
            "execution_unit_key": unit_key,
            "logical_execution_index": unit.get("logical_execution_index", ""),
            "net_id": unit.get("net_id", ""),
            "iteration": unit.get("iteration", ""),
            "is_internal_network_candidate": unit.get("is_internal_network_candidate", ""),
            "dispatch_rows_total": unit["dispatch_rows_total"],
            "matched_dispatch_rows": unit["matched_dispatch_rows"],
            "kernel_aligned_rows": unit["kernel_aligned_rows"],
            "topdown_primitives_total": len(unit_topdown),
            "topdown_primitives_mapped": len(unit_mapped),
            "topdown_mode": topology_mode_by_unit.get(unit_key, "missing_topdown"),
        })

    gap_summary = {
        "dispatch_rows_total": len(dispatch_rows),
        "topdown_rows_total": len(topdown_rows),
        "execution_units_total": len(execution_units),
        "dispatch_rows_without_topdown": dispatch_without_topdown,
        "topdown_rows_without_dispatch": unmapped_topdown_rows,
        "multi_kernel_primitives": multi_kernel_primitives,
        "inference_scoped_topdown_units": sum(1 for mode in topology_mode_by_unit.values() if mode == "inference_scoped"),
        "replicated_network_scoped_units": sum(1 for mode in topology_mode_by_unit.values() if mode == "replicated_from_network_topology"),
    }

    return execution_unit_rows, component_group_nodes + op_type_nodes + layer_nodes + primitive_nodes, edges, gap_summary


def write_gap_report(path: Path, gap_summary: Dict[str, object], execution_unit_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Graph Join Preparation Report",
        "",
        "## Summary",
        "",
        f"- Dispatch rows total: {gap_summary['dispatch_rows_total']}",
        f"- Top-down rows total after normalization: {gap_summary['topdown_rows_total']}",
        f"- Execution units total: {gap_summary['execution_units_total']}",
        f"- Dispatch rows without top-down primitive match: {gap_summary['dispatch_rows_without_topdown']}",
        f"- Top-down primitive rows without dispatch match: {gap_summary['topdown_rows_without_dispatch']}",
        f"- Primitive nodes with more than one kernel entry: {gap_summary['multi_kernel_primitives']}",
        f"- Execution units with inference-scoped top-down rows: {gap_summary['inference_scoped_topdown_units']}",
        f"- Execution units using replicated network-scoped topology: {gap_summary['replicated_network_scoped_units']}",
        "",
        "## Execution Units",
        "",
    ]
    for row in execution_unit_rows:
        lines.append(
            f"- {row['execution_unit_key']}: net_id={row['net_id']}, iteration={row['iteration']}, "
            f"dispatch_rows={row['dispatch_rows_total']}, topdown_primitives={row['topdown_primitives_total']}, "
            f"mapped_primitives={row['topdown_primitives_mapped']}, topdown_mode={row['topdown_mode']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_execution_unit_bundles(
    output_dir: Path,
    bundle_name: str,
    execution_unit_rows: List[Dict[str, object]],
    hotspot_rows: List[Dict[str, object]],
    hierarchy_rows: List[Dict[str, object]],
    graph_nodes: List[Dict[str, object]],
    graph_edges: List[Dict[str, object]],
) -> None:
    for unit_row in execution_unit_rows:
        execution_unit_key = str(unit_row["execution_unit_key"])
        unit_dir = output_dir / execution_unit_key
        unit_dir.mkdir(parents=True, exist_ok=True)

        unit_hotspots = [row for row in hotspot_rows if str(row.get("execution_unit_key", "")) == execution_unit_key]
        unit_hierarchy = [row for row in hierarchy_rows if str(row.get("execution_unit_key", "")) == execution_unit_key]
        unit_graph_nodes = [row for row in graph_nodes if str(row.get("execution_unit_key", "")) == execution_unit_key]
        unit_graph_edges = [row for row in graph_edges if str(row.get("execution_unit_key", "")) == execution_unit_key]

        write_csv(
            unit_dir / f"{bundle_name}_{execution_unit_key}_hotspot_table.csv",
            unit_hotspots,
            HOTSPOT_FIELDS,
        )
        write_csv(
            unit_dir / f"{bundle_name}_{execution_unit_key}_hierarchy_rows.csv",
            unit_hierarchy,
            HIERARCHY_FIELDS,
        )
        write_csv(
            unit_dir / f"{bundle_name}_{execution_unit_key}_graph_nodes.csv",
            unit_graph_nodes,
            GRAPH_NODE_FIELDS,
        )
        write_csv(
            unit_dir / f"{bundle_name}_{execution_unit_key}_graph_edges.csv",
            unit_graph_edges,
            GRAPH_EDGE_FIELDS,
        )

        summary = {
            "execution_unit_key": execution_unit_key,
            "logical_execution_index": unit_row.get("logical_execution_index", ""),
            "net_id": unit_row.get("net_id", ""),
            "iteration": unit_row.get("iteration", ""),
            "is_internal_network_candidate": unit_row.get("is_internal_network_candidate", ""),
            "dispatch_rows_total": unit_row.get("dispatch_rows_total", 0),
            "matched_dispatch_rows": unit_row.get("matched_dispatch_rows", 0),
            "kernel_aligned_rows": unit_row.get("kernel_aligned_rows", 0),
            "topdown_primitives_total": unit_row.get("topdown_primitives_total", 0),
            "topdown_primitives_mapped": unit_row.get("topdown_primitives_mapped", 0),
            "topdown_mode": unit_row.get("topdown_mode", ""),
            "hotspot_rows": len(unit_hotspots),
            "hierarchy_rows": len(unit_hierarchy),
            "graph_nodes": len(unit_graph_nodes),
            "graph_edges": len(unit_graph_edges),
        }
        (unit_dir / f"{bundle_name}_{execution_unit_key}_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare graph-oriented join artifacts from dispatch and top-down dumps.")
    parser.add_argument("--dispatch-join", required=True, help="Path to dispatch_gtpin_kernel_metrics_join.csv")
    parser.add_argument("--topdown", required=True, help="Path to ov_topdown_primitive_rows*.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for graph-prep outputs")
    parser.add_argument("--bundle-name", default="graph_join_bundle", help="Output filename prefix")
    args = parser.parse_args()

    dispatch_path = Path(args.dispatch_join)
    topdown_path = Path(args.topdown)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch_rows = normalize_dispatch_rows(read_csv_rows(dispatch_path))
    topdown_rows = normalize_topdown_rows(read_csv_rows(topdown_path))

    execution_units = build_execution_units(dispatch_rows)
    expanded_topdown_rows, topology_mode_by_unit = expand_topdown_by_execution_unit(topdown_rows, execution_units)
    execution_unit_rows, _legacy_graph_nodes, _legacy_graph_edges, gap_summary = aggregate_graph_rows(
        dispatch_rows=dispatch_rows,
        topdown_rows=expanded_topdown_rows,
        execution_units=execution_units,
        topology_mode_by_unit=topology_mode_by_unit,
    )
    matched_hierarchy_rows = build_matched_hierarchy_rows(expanded_topdown_rows, dispatch_rows, execution_units)
    hotspot_rows = build_hotspot_rows(matched_hierarchy_rows, dispatch_rows, execution_units)
    graph_nodes, graph_edges = build_graph_structure_rows(hotspot_rows)

    bundle_prefix = output_dir / args.bundle_name
    write_csv(
        bundle_prefix.with_name(f"{args.bundle_name}_execution_units.csv"),
        execution_unit_rows,
        [
            "execution_unit_key", "logical_execution_index", "net_id", "iteration", "is_internal_network_candidate",
            "dispatch_rows_total", "matched_dispatch_rows",
            "kernel_aligned_rows", "topdown_primitives_total", "topdown_primitives_mapped", "topdown_mode",
        ],
    )
    write_csv(
        bundle_prefix.with_name(f"{args.bundle_name}_hotspot_table.csv"),
        hotspot_rows,
        HOTSPOT_FIELDS,
    )
    write_csv(
        bundle_prefix.with_name(f"{args.bundle_name}_hierarchy_rows.csv"),
        matched_hierarchy_rows,
        HIERARCHY_FIELDS,
    )
    write_csv(
        bundle_prefix.with_name(f"{args.bundle_name}_graph_nodes.csv"),
        graph_nodes,
        GRAPH_NODE_FIELDS,
    )
    write_csv(
        bundle_prefix.with_name(f"{args.bundle_name}_graph_edges.csv"),
        graph_edges,
        GRAPH_EDGE_FIELDS,
    )

    bundle_json = {
        "dispatch_join": str(dispatch_path),
        "topdown": str(topdown_path),
        "gap_summary": gap_summary,
        "execution_units": execution_unit_rows,
        "hotspot_rows_total": len(hotspot_rows),
        "hierarchy_rows_total": len(matched_hierarchy_rows),
        "graph_nodes_total": len(graph_nodes),
        "graph_edges_total": len(graph_edges),
    }
    bundle_prefix.with_name(f"{args.bundle_name}_summary.json").write_text(
        json.dumps(bundle_json, indent=2),
        encoding="utf-8",
    )
    write_gap_report(bundle_prefix.with_name(f"{args.bundle_name}_report.md"), gap_summary, execution_unit_rows)
    write_execution_unit_bundles(
        output_dir=output_dir,
        bundle_name=args.bundle_name,
        execution_unit_rows=execution_unit_rows,
        hotspot_rows=hotspot_rows,
        hierarchy_rows=matched_hierarchy_rows,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
    )


if __name__ == "__main__":
    main()
