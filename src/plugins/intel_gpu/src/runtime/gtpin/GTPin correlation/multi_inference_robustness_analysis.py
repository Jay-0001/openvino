#!/usr/bin/env python3
"""
Tool-agnostic multi-inference robustness analysis for OpenVINO dispatch dumps and
GTPin text outputs.

This script generalizes the first join so downstream
stages can consume:
- correlation fields shared across all tools
- generic primary-metric fields
- tool-specific metric columns
- a serialized payload of any additional parsed metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


MISSING_OV_KERNEL_ENTRY_LABEL = "__missing_ov_kernel_entry__"

STANDARD_JOIN_FIELDS = [
    "model_label",
    "tool_kind",
    "dispatch_id",
    "ov_match",
    "kernel_alignment_status",
    "kernel_entry_alignment_verified",
    "ov_kernel_entry_present",
    "net_id",
    "net_id_num",
    "iteration",
    "iteration_num",
    "execution_unit_key",
    "is_internal_network_candidate",
    "dispatch_index",
    "primitive_id",
    "primitive_type",
    "implementation",
    "kernel_entry",
    "kernel_entry_raw",
    "batch_hash",
    "gtpin_kernel",
    "gtpin_unique_name",
    "gtpin_extended_name",
    "execution_descriptor",
    "kernel_name_match",
    "pointer_coverage_match",
    "gtpin_source_kind",
    "primary_metric_name",
    "primary_metric_total",
    "primary_metric_avg",
    "primary_metric_unit",
    "gtpin_input_pointers",
    "gtpin_output_pointers",
    "gtpin_all_pointers",
    "gtpin_raw_pointer_records",
    "gtpin_metric_fields_json",
]

PERFORMANCE_METRIC_FIELDS = [
    "gtpin_invocation_count",
    "gtpin_total_execution_cycles",
    "gtpin_avg_execution_cycles_per_invocation",
]

MEMORY_METRIC_FIELDS = [
    "gtpin_total_memory_ops",
    "gtpin_estimated_total_bytes",
    "gtpin_write_dominance_pct",
    "gtpin_estimated_bytes_per_mem_op",
    "gtpin_reads",
    "gtpin_writes",
    "gtpin_atomics",
]

STREAMLINED_JOIN_FIELDS = [
    "model_label",
    "tool_kind",
    "dispatch_id",
    "kernel_entry",
    "ov_match",
    "kernel_alignment_status",
    "kernel_entry_alignment_verified",
    "primitive_id",
    "primitive_type",
    "implementation",
    "net_id",
    "iteration",
    "execution_unit_key",
    "is_internal_network_candidate",
    "gtpin_kernel",
    "gtpin_source_kind",
    "primary_metric_name",
    "primary_metric_total",
    "primary_metric_avg",
    "primary_metric_unit",
    "execution_descriptor",
    "pointer_coverage_match",
    "ov_kernel_entry_present",
    "gtpin_metric_fields_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join OpenVINO dispatch dumps with tool-agnostic GTPin profiling text outputs."
    )
    parser.add_argument(
        "-DispatchCsv",
        "--DispatchCsv",
        "--dispatch-csv",
        dest="dispatch_csv",
        required=True,
        help="Path to the OpenVINO dispatch CSV.",
    )
    parser.add_argument(
        "-GtpinDump",
        "--GtpinDump",
        "--gtpin-dump",
        "--KernelArgDump",
        "--kernel-arg-dump",
        "--GTPinProfile",
        "--gtpin-profile",
        dest="gtpin_dump",
        required=True,
        help="Path to the normalized GTPin dump.",
    )
    parser.add_argument(
        "-GtpinInputFormat",
        "--GtpinInputFormat",
        "--gtpin-input-format",
        dest="gtpin_input_format",
        default="auto",
        choices=("auto", "raw_dispatch_dump", "exec_profile", "memory_axis"),
        help="Input-format hint for the GTPin dump.",
    )
    parser.add_argument(
        "--tool-kind",
        dest="tool_kind",
        default="auto",
        choices=("auto", "performance", "memory", "raw_dispatch"),
        help="Optional explicit tool-kind label for the invocation.",
    )
    parser.add_argument(
        "-OutputDir",
        "--OutputDir",
        "--output-dir",
        dest="output_dir",
        default="",
        help="Optional output directory.",
    )
    parser.add_argument(
        "-ModelLabel",
        "--ModelLabel",
        "--model-label",
        dest="model_label",
        default="unknown_model",
        help="Model label used in generated outputs.",
    )
    return parser.parse_args()


def to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip().rstrip("%")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def join_unique_values(values: Iterable[object]) -> str:
    items = sorted({str(value).strip() for value in values if str(value).strip()})
    return ";".join(items)


def get_address_entries(value: str) -> List[str]:
    if not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def get_address_identity(entry: str) -> str:
    parts = str(entry).split(":")
    if not parts:
        return ""
    return parts[-1].strip()


def get_gtpin_pointer_list(values: Sequence[str]) -> List[str]:
    pointers: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        pointers.append(text.removeprefix("0x").upper())
    return pointers


def test_pointer_coverage(expected: Sequence[str], actual_set: set[str]) -> bool:
    return all(pointer in actual_set for pointer in expected)


def get_execution_unit_key(net_id: int, iteration: int) -> str:
    return f"n{net_id}_i{iteration}"


@dataclass
class GTPinDispatchRecord:
    dispatch_id: int
    source_kind: str
    tool_kind: str
    kernel: str = ""
    unique_name: str = ""
    extended_name: str = ""
    execution_descriptor: str = ""
    pointer_values: List[str] = field(default_factory=list)
    input_pointer_values: List[str] = field(default_factory=list)
    output_pointer_values: List[str] = field(default_factory=list)
    arg_ordinals: List[str] = field(default_factory=list)
    raw_pointer_records: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)

    def metric_json(self) -> str:
        return json.dumps(self.metrics, sort_keys=True, separators=(",", ":"))


@dataclass
class OVDispatchRow:
    net_id: str
    net_id_num: int
    iteration: str
    iteration_num: int
    execution_unit_key: str
    is_internal_network_candidate: str
    dispatch_index: str
    dispatch_index_num: int
    global_dispatch_id: int
    primitive_id: str
    primitive_type: str
    implementation: str
    kernel_index: str
    kernel_index_num: int
    kernel_entry: str
    kernel_entry_effective: str
    has_kernel_entry: str
    batch_hash: str
    input_arg_addresses: str
    output_arg_addresses: str
    output_memory_addresses: str


def infer_input_format(path: Path, preferred_format: str) -> str:
    if preferred_format != "auto":
        return preferred_format

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _, line in zip(range(40), handle):
            if re.search(r"^InvocationCount:\s*\d+", line):
                return "exec_profile"
            if re.search(r"^TotalMemoryOps:\s*\d+", line):
                return "memory_axis"
            if re.search(r"^### Memory-axis kernel correlation profile generated by GTPin ###", line):
                return "memory_axis"
            if re.search(r"^UniqueName:\s*", line) or re.search(r"^ExtendedName:\s*", line) or re.search(r"^ExecutionDescriptor:\s*", line):
                return "raw_dispatch_dump"

    return "raw_dispatch_dump"


def resolved_tool_kind(format_name: str, requested_tool_kind: str) -> str:
    if requested_tool_kind != "auto":
        return requested_tool_kind
    if format_name == "exec_profile":
        return "performance"
    if format_name == "memory_axis":
        return "memory"
    return "raw_dispatch"


def append_pointer(record: GTPinDispatchRecord, pointer_value: str, current_role: str) -> None:
    ordinal = ""
    if len(record.arg_ordinals) > len(record.pointer_values):
        ordinal = record.arg_ordinals[len(record.pointer_values)]
    record.pointer_values.append(pointer_value)
    record.raw_pointer_records.append(f"{ordinal}:{pointer_value}".strip(":"))
    if current_role == "INPUT":
        record.input_pointer_values.append(pointer_value)
    elif current_role in {"OUTPUT", "INOUT"}:
        record.output_pointer_values.append(pointer_value)


def parse_raw_dispatch_dump(path: Path) -> List[GTPinDispatchRecord]:
    dispatches: List[GTPinDispatchRecord] = []
    current: GTPinDispatchRecord | None = None
    current_role = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.match(r"^DispatchId:\s*(\d+)", line)
            if match:
                if current is not None:
                    dispatches.append(current)
                current = GTPinDispatchRecord(
                    dispatch_id=int(match.group(1)),
                    source_kind="raw_dispatch_dump",
                    tool_kind="raw_dispatch",
                )
                current_role = ""
                continue

            if current is None:
                continue

            if match := re.match(r"^Kernel:\s*(.+)$", line):
                current.kernel = match.group(1).strip()
                continue
            if match := re.match(r"^UniqueName:\s*(.+)$", line):
                current.unique_name = match.group(1).strip()
                continue
            if match := re.match(r"^ExtendedName:\s*(.+)$", line):
                current.extended_name = match.group(1).strip()
                continue
            if match := re.match(r"^ExecutionDescriptor:\s*(.+)$", line):
                current.execution_descriptor = match.group(1).strip()
                continue
            if match := re.match(r"^\s*ArgOrdinal=(\d+).*Type=arg_bypointer.*Role=([A-Z]+)", line):
                current.arg_ordinals.append(match.group(1))
                current_role = match.group(2)
                continue
            if match := re.match(r"^\s*PointerValue=(0x[0-9A-Fa-f]+)", line):
                append_pointer(current, match.group(1), current_role)
                continue

    if current is not None:
        dispatches.append(current)

    return dispatches


def parse_exec_profile(path: Path) -> List[GTPinDispatchRecord]:
    dispatches: List[GTPinDispatchRecord] = []
    current: GTPinDispatchRecord | None = None
    current_role = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.match(r"^DispatchId:\s*(\d+)", line)
            if match:
                if current is not None:
                    dispatches.append(current)
                current = GTPinDispatchRecord(
                    dispatch_id=int(match.group(1)),
                    source_kind="exec_profile",
                    tool_kind="performance",
                    metrics={
                        "gtpin_invocation_count": 0,
                        "gtpin_total_execution_cycles": 0,
                        "gtpin_avg_execution_cycles_per_invocation": 0,
                    },
                )
                current_role = ""
                continue

            if current is None:
                continue

            if match := re.match(r"^Kernel:\s*(.+)$", line):
                current.kernel = match.group(1).strip()
                continue

            if match := re.match(
                r"^InvocationCount:\s*(\d+)\s+TotalExecutionCycles:\s*(\d+)\s+AvgExecutionCyclesPerInvocation:\s*(\d+)",
                line,
            ):
                invocation_count = int(match.group(1))
                total_cycles = int(match.group(2))
                avg_cycles = int(match.group(3))
                current.metrics.update({
                    "gtpin_invocation_count": invocation_count,
                    "gtpin_total_execution_cycles": total_cycles,
                    "gtpin_avg_execution_cycles_per_invocation": avg_cycles,
                })
                current.execution_descriptor = (
                    f"InvocationCount={invocation_count};"
                    f"TotalExecutionCycles={total_cycles};"
                    f"AvgExecutionCyclesPerInvocation={avg_cycles}"
                )
                continue

            if match := re.match(r"^\s*ArgOrdinal=(\d+).*Role=([A-Z]+)", line):
                current.arg_ordinals.append(match.group(1))
                current_role = match.group(2)
                continue
            if match := re.match(r"^\s*RawValue=(0x[0-9A-Fa-f]+)\s+PointerValue=(0x[0-9A-Fa-f]+)", line):
                append_pointer(current, match.group(2), current_role)
                continue
            if match := re.match(r"^\s*PointerValue=(0x[0-9A-Fa-f]+)", line):
                append_pointer(current, match.group(1), current_role)
                continue

    if current is not None:
        dispatches.append(current)

    return dispatches


def parse_memory_axis(path: Path) -> List[GTPinDispatchRecord]:
    dispatches: List[GTPinDispatchRecord] = []
    current: GTPinDispatchRecord | None = None
    current_role = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if re.match(r"^-{20,}$", line):
                continue

            if match := re.match(r"^DispatchId:\s*(\d+)", line):
                if current is not None:
                    if not current.execution_descriptor:
                        current.execution_descriptor = (
                            f"TotalMemoryOps={current.metrics['gtpin_total_memory_ops']};"
                            f"EstimatedTotalBytes={current.metrics['gtpin_estimated_total_bytes']};"
                            f"WriteDominancePct={current.metrics['gtpin_write_dominance_pct']};"
                            f"EstimatedBytesPerMemOp={current.metrics['gtpin_estimated_bytes_per_mem_op']};"
                            f"Reads={current.metrics['gtpin_reads']};"
                            f"Writes={current.metrics['gtpin_writes']};"
                            f"Atomics={current.metrics['gtpin_atomics']}"
                        )
                    dispatches.append(current)
                current = GTPinDispatchRecord(
                    dispatch_id=int(match.group(1)),
                    source_kind="memory_axis",
                    tool_kind="memory",
                    metrics={
                        "gtpin_total_memory_ops": 0,
                        "gtpin_estimated_total_bytes": 0,
                        "gtpin_write_dominance_pct": 0.0,
                        "gtpin_estimated_bytes_per_mem_op": 0.0,
                        "gtpin_reads": 0,
                        "gtpin_writes": 0,
                        "gtpin_atomics": 0,
                    },
                )
                current_role = ""
                continue

            if current is None:
                continue

            if match := re.match(r"^Kernel:\s*(.+)$", line):
                current.kernel = match.group(1).strip()
                continue

            if match := re.match(r"^TotalMemoryOps:\s*(\d+)\s+EstimatedTotalBytes:\s*(\d+)", line):
                current.metrics["gtpin_total_memory_ops"] = int(match.group(1))
                current.metrics["gtpin_estimated_total_bytes"] = int(match.group(2))
                continue

            if match := re.match(r"^WriteDominance:\s*([0-9.]+)%\s+EstimatedBytesPerMemOp:\s*([0-9.]+)", line):
                current.metrics["gtpin_write_dominance_pct"] = float(match.group(1))
                current.metrics["gtpin_estimated_bytes_per_mem_op"] = float(match.group(2))
                continue

            if match := re.match(r"^AccessBreakdown:\s*Reads=(\d+)\s+Writes=(\d+)\s+Atomics=(\d+)", line):
                current.metrics["gtpin_reads"] = int(match.group(1))
                current.metrics["gtpin_writes"] = int(match.group(2))
                current.metrics["gtpin_atomics"] = int(match.group(3))
                continue

            if match := re.match(r"^\s*ArgOrdinal=(\d+).*Role=([A-Z]+)", line):
                current.arg_ordinals.append(match.group(1))
                current_role = match.group(2)
                continue

            if match := re.match(r"^\s*RawValue=(0x[0-9A-Fa-f]+)\s+PointerValue=(0x[0-9A-Fa-f]+)", line):
                append_pointer(current, match.group(2), current_role)
                continue

            if match := re.match(r"^\s*PointerValue=(0x[0-9A-Fa-f]+)", line):
                append_pointer(current, match.group(1), current_role)
                continue

    if current is not None:
        if not current.execution_descriptor:
            current.execution_descriptor = (
                f"TotalMemoryOps={current.metrics['gtpin_total_memory_ops']};"
                f"EstimatedTotalBytes={current.metrics['gtpin_estimated_total_bytes']};"
                f"WriteDominancePct={current.metrics['gtpin_write_dominance_pct']};"
                f"EstimatedBytesPerMemOp={current.metrics['gtpin_estimated_bytes_per_mem_op']};"
                f"Reads={current.metrics['gtpin_reads']};"
                f"Writes={current.metrics['gtpin_writes']};"
                f"Atomics={current.metrics['gtpin_atomics']}"
            )
        dispatches.append(current)

    return dispatches


def parse_gtpin_dispatch_rows(path: Path, preferred_format: str, requested_tool_kind: str) -> tuple[str, str, List[GTPinDispatchRecord]]:
    resolved_format = infer_input_format(path, preferred_format)
    parser_map = {
        "raw_dispatch_dump": parse_raw_dispatch_dump,
        "exec_profile": parse_exec_profile,
        "memory_axis": parse_memory_axis,
    }
    if resolved_format not in parser_map:
        raise ValueError(f"Unsupported GTPin input format: {resolved_format}")

    rows = parser_map[resolved_format](path)
    tool_kind = resolved_tool_kind(resolved_format, requested_tool_kind)
    for record in rows:
        record.tool_kind = tool_kind
    rows.sort(key=lambda item: item.dispatch_id)
    return resolved_format, tool_kind, rows


def parse_dispatch_csv(path: Path) -> List[OVDispatchRow]:
    rows: List[OVDispatchRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            global_dispatch_id = to_int(row.get("global_dispatch_id") or row.get("dispatch_index"), default=0)
            net_id_num = to_int(row.get("net_id"), default=0)
            iteration_num = to_int(row.get("iteration"), default=0)
            kernel_entry = row.get("kernel_entry", "")
            has_kernel_entry = bool_text(bool(str(kernel_entry).strip()))
            rows.append(
                OVDispatchRow(
                    net_id=row.get("net_id", ""),
                    net_id_num=net_id_num,
                    iteration=row.get("iteration", ""),
                    iteration_num=iteration_num,
                    execution_unit_key=get_execution_unit_key(net_id_num, iteration_num),
                    is_internal_network_candidate=bool_text(net_id_num == 0),
                    dispatch_index=row.get("dispatch_index", ""),
                    dispatch_index_num=to_int(row.get("dispatch_index"), default=0),
                    global_dispatch_id=global_dispatch_id,
                    primitive_id=row.get("primitive_id", ""),
                    primitive_type=row.get("primitive_type", ""),
                    implementation=row.get("implementation", ""),
                    kernel_index=row.get("kernel_index", ""),
                    kernel_index_num=to_int(row.get("kernel_index"), default=0),
                    kernel_entry=kernel_entry,
                    kernel_entry_effective=kernel_entry if str(kernel_entry).strip() else MISSING_OV_KERNEL_ENTRY_LABEL,
                    has_kernel_entry=has_kernel_entry,
                    batch_hash=row.get("batch_hash", ""),
                    input_arg_addresses=row.get("input_arg_addresses", ""),
                    output_arg_addresses=row.get("output_arg_addresses", ""),
                    output_memory_addresses=row.get("output_memory_addresses", ""),
                )
            )
    rows.sort(key=lambda item: item.global_dispatch_id)
    return rows


def metric_policy(tool_kind: str, metrics: Dict[str, object]) -> tuple[str, str, str, str]:
    if tool_kind == "performance":
        return (
            "gtpin_total_execution_cycles",
            str(to_int(metrics.get("gtpin_total_execution_cycles"), default=0)),
            str(to_int(metrics.get("gtpin_avg_execution_cycles_per_invocation"), default=0)),
            "cycles",
        )
    if tool_kind == "memory":
        return (
            "gtpin_estimated_total_bytes",
            str(to_int(metrics.get("gtpin_estimated_total_bytes"), default=0)),
            str(to_float(metrics.get("gtpin_estimated_bytes_per_mem_op"), default=0.0)),
            "estimated_bytes",
        )
    return ("", "", "", "")


def format_join_row(model_label: str, ov_row: OVDispatchRow | None, gtpin_row: GTPinDispatchRecord) -> Dict[str, object]:
    ov_input_pointers: List[str] = []
    ov_output_pointers: List[str] = []
    gtpin_input_pointers = get_gtpin_pointer_list(gtpin_row.input_pointer_values)
    gtpin_output_pointers = get_gtpin_pointer_list(gtpin_row.output_pointer_values)
    gtpin_all_pointers = get_gtpin_pointer_list(gtpin_row.pointer_values)
    gtpin_all_pointer_set = set(gtpin_all_pointers)

    pointer_coverage = "false"
    kernel_name_match = "false"
    ov_match = "missing_ov_dispatch"
    kernel_alignment_status = "missing_ov_dispatch"
    kernel_entry_alignment_verified = "false"

    if ov_row is not None:
        ov_input_pointers = [
            get_address_identity(entry).upper()
            for entry in get_address_entries(ov_row.input_arg_addresses)
            if get_address_identity(entry).strip()
        ]
        ov_output_pointers = [
            get_address_identity(entry).upper()
            for entry in get_address_entries(ov_row.output_arg_addresses)
            if get_address_identity(entry).strip()
        ]
        input_covered = test_pointer_coverage(ov_input_pointers, gtpin_all_pointer_set)
        output_covered = test_pointer_coverage(ov_output_pointers, gtpin_all_pointer_set)
        pointer_coverage = bool_text(input_covered and output_covered)

        if ov_row.has_kernel_entry == "true":
            ov_match = "matched"
            kernel_name_match = bool_text(ov_row.kernel_entry == gtpin_row.kernel)
            if kernel_name_match == "true":
                kernel_alignment_status = "aligned"
                kernel_entry_alignment_verified = "true"
            else:
                kernel_alignment_status = "name_mismatch"
        else:
            ov_match = "missing_ov_kernel_entry"
            kernel_alignment_status = "ov_kernel_entry_missing"

    primary_metric_name, primary_metric_total, primary_metric_avg, primary_metric_unit = metric_policy(
        gtpin_row.tool_kind,
        gtpin_row.metrics,
    )

    joined_row: Dict[str, object] = {
        "model_label": model_label,
        "tool_kind": gtpin_row.tool_kind,
        "dispatch_id": gtpin_row.dispatch_id,
        "ov_match": ov_match,
        "kernel_alignment_status": kernel_alignment_status,
        "kernel_entry_alignment_verified": kernel_entry_alignment_verified,
        "ov_kernel_entry_present": ov_row.has_kernel_entry if ov_row is not None else "false",
        "net_id": ov_row.net_id if ov_row is not None else "",
        "net_id_num": ov_row.net_id_num if ov_row is not None else -1,
        "iteration": ov_row.iteration if ov_row is not None else "",
        "iteration_num": ov_row.iteration_num if ov_row is not None else -1,
        "execution_unit_key": ov_row.execution_unit_key if ov_row is not None else "",
        "is_internal_network_candidate": ov_row.is_internal_network_candidate if ov_row is not None else "false",
        "dispatch_index": ov_row.dispatch_index if ov_row is not None else "",
        "primitive_id": ov_row.primitive_id if ov_row is not None else "",
        "primitive_type": ov_row.primitive_type if ov_row is not None else "",
        "implementation": ov_row.implementation if ov_row is not None else "",
        "kernel_entry": ov_row.kernel_entry_effective if ov_row is not None else "",
        "kernel_entry_raw": ov_row.kernel_entry if ov_row is not None else "",
        "batch_hash": ov_row.batch_hash if ov_row is not None else "",
        "gtpin_kernel": gtpin_row.kernel,
        "gtpin_unique_name": gtpin_row.unique_name,
        "gtpin_extended_name": gtpin_row.extended_name,
        "execution_descriptor": gtpin_row.execution_descriptor,
        "kernel_name_match": kernel_name_match,
        "pointer_coverage_match": pointer_coverage,
        "gtpin_source_kind": gtpin_row.source_kind,
        "primary_metric_name": primary_metric_name,
        "primary_metric_total": primary_metric_total,
        "primary_metric_avg": primary_metric_avg,
        "primary_metric_unit": primary_metric_unit,
        "gtpin_input_pointers": ";".join(gtpin_input_pointers),
        "gtpin_output_pointers": ";".join(gtpin_output_pointers),
        "gtpin_all_pointers": ";".join(gtpin_all_pointers),
        "gtpin_raw_pointer_records": ";".join(gtpin_row.raw_pointer_records),
        "gtpin_metric_fields_json": gtpin_row.metric_json(),
        "gtpin_invocation_count": "",
        "gtpin_total_execution_cycles": "",
        "gtpin_avg_execution_cycles_per_invocation": "",
        "gtpin_total_memory_ops": "",
        "gtpin_estimated_total_bytes": "",
        "gtpin_write_dominance_pct": "",
        "gtpin_estimated_bytes_per_mem_op": "",
        "gtpin_reads": "",
        "gtpin_writes": "",
        "gtpin_atomics": "",
    }
    for key, value in gtpin_row.metrics.items():
        joined_row[key] = value
    return joined_row


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize_execution_units(model_label: str, dispatch_rows: Sequence[OVDispatchRow], joined_rows: Sequence[Dict[str, object]], ov_only_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    execution_unit_order: Dict[str, int] = {}
    steady_units = sorted(
        {row.execution_unit_key for row in dispatch_rows if row.net_id_num != 0},
        key=lambda key: (
            next(row.net_id_num for row in dispatch_rows if row.execution_unit_key == key),
            next(row.iteration_num for row in dispatch_rows if row.execution_unit_key == key),
            key,
        ),
    )
    for logical_index, unit_key in enumerate(steady_units):
        execution_unit_order[unit_key] = logical_index

    grouped: Dict[str, List[OVDispatchRow]] = defaultdict(list)
    for row in dispatch_rows:
        grouped[row.execution_unit_key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for unit_key in sorted(grouped, key=lambda key: (grouped[key][0].global_dispatch_id, key)):
        group = sorted(grouped[unit_key], key=lambda item: item.global_dispatch_id)
        first = group[0]
        matched_rows = [row for row in joined_rows if row["ov_match"] == "matched" and row["execution_unit_key"] == unit_key]
        missing_kernel_rows = [row for row in joined_rows if row["ov_match"] == "missing_ov_kernel_entry" and row["execution_unit_key"] == unit_key]
        missing_gtpin_rows = [row for row in ov_only_rows if row["execution_unit_key"] == unit_key]
        summary_rows.append({
            "model_label": model_label,
            "execution_unit_key": unit_key,
            "net_id": first.net_id,
            "iteration": first.iteration,
            "is_internal_network_candidate": first.is_internal_network_candidate,
            "logical_execution_index": execution_unit_order.get(unit_key, ""),
            "ov_dispatch_rows": len(group),
            "matched_gtpin_rows": len(matched_rows),
            "missing_ov_kernel_entry_rows": len(missing_kernel_rows),
            "missing_gtpin_rows": len(missing_gtpin_rows),
            "distinct_ov_kernels": len({row.kernel_entry_effective for row in group}),
            "distinct_matched_gtpin_kernels": len({str(row["gtpin_kernel"]) for row in matched_rows}),
            "first_dispatch_id": first.global_dispatch_id,
            "last_dispatch_id": group[-1].global_dispatch_id,
        })
    return summary_rows


def summarize_networks(model_label: str, dispatch_rows: Sequence[OVDispatchRow], execution_unit_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[OVDispatchRow]] = defaultdict(list)
    for row in dispatch_rows:
        grouped[row.net_id_num].append(row)

    results: List[Dict[str, object]] = []
    for net_id_num in sorted(grouped):
        group = sorted(grouped[net_id_num], key=lambda item: item.global_dispatch_id)
        unit_rows = sorted(
            [row for row in execution_unit_rows if to_int(row["net_id"], default=-1) == net_id_num],
            key=lambda item: to_int(item["first_dispatch_id"], default=0),
        )
        rows_per_unit = [to_int(row["ov_dispatch_rows"], default=0) for row in unit_rows]
        if net_id_num == 0:
            interpretation = "internal_setup_network_candidate"
        elif len(unit_rows) > 1:
            interpretation = "repeated_execution_network"
        else:
            interpretation = "single_execution_network"
        results.append({
            "model_label": model_label,
            "net_id": group[0].net_id,
            "is_internal_network_candidate": bool_text(net_id_num == 0),
            "execution_unit_count": len(unit_rows),
            "rows_total": len(group),
            "rows_per_execution_unit": ";".join(f"{row['iteration']}:{row['ov_dispatch_rows']}" for row in unit_rows),
            "distinct_row_counts_per_execution_unit": join_unique_values(rows_per_unit),
            "distinct_kernel_entries": len({row.kernel_entry_effective for row in group}),
            "first_dispatch_id": group[0].global_dispatch_id,
            "last_dispatch_id": group[-1].global_dispatch_id,
            "interpretation": interpretation,
        })
    return results


def summarize_kernels(
    model_label: str,
    dispatch_rows: Sequence[OVDispatchRow],
    gtpin_rows: Sequence[GTPinDispatchRecord],
    joined_rows: Sequence[Dict[str, object]],
    execution_unit_rows: Sequence[Dict[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    execution_unit_order = {
        str(row["execution_unit_key"]): row.get("logical_execution_index", "")
        for row in execution_unit_rows
    }
    all_kernel_names = sorted(
        {
            *(row.kernel_entry_effective for row in dispatch_rows),
            *(row.kernel for row in gtpin_rows),
        }
    )
    kernel_execution_unit_rows: List[Dict[str, object]] = []
    kernel_summary_rows: List[Dict[str, object]] = []

    for kernel_name in all_kernel_names:
        ov_kernel_rows = sorted(
            [row for row in dispatch_rows if row.kernel_entry_effective == kernel_name],
            key=lambda item: item.global_dispatch_id,
        )
        gtpin_kernel_rows = sorted(
            [row for row in gtpin_rows if row.kernel == kernel_name],
            key=lambda item: item.dispatch_id,
        )
        direct_joined_rows = sorted(
            [row for row in joined_rows if row["gtpin_kernel"] == kernel_name and row["ov_match"] != "missing_ov_dispatch"],
            key=lambda item: to_int(item["dispatch_id"], default=0),
        )
        matched_kernel_rows = sorted(
            [row for row in joined_rows if row["gtpin_kernel"] == kernel_name and row["ov_match"] == "matched" and row["kernel_name_match"] == "true"],
            key=lambda item: to_int(item["dispatch_id"], default=0),
        )
        missing_kernel_entry_rows = sorted(
            [row for row in joined_rows if row["gtpin_kernel"] == kernel_name and row["ov_match"] == "missing_ov_kernel_entry"],
            key=lambda item: to_int(item["dispatch_id"], default=0),
        )

        grouped_units: Dict[str, List[OVDispatchRow]] = defaultdict(list)
        for row in ov_kernel_rows:
            grouped_units[row.execution_unit_key].append(row)

        for unit_key in sorted(grouped_units, key=lambda key: grouped_units[key][0].global_dispatch_id):
            group = sorted(grouped_units[unit_key], key=lambda item: item.global_dispatch_id)
            first = group[0]
            kernel_execution_unit_rows.append({
                "model_label": model_label,
                "kernel_entry": kernel_name,
                "execution_unit_key": unit_key,
                "net_id": first.net_id,
                "iteration": first.iteration,
                "is_internal_network_candidate": first.is_internal_network_candidate,
                "logical_execution_index": execution_unit_order.get(unit_key, ""),
                "ov_dispatch_rows": len(group),
                "matched_gtpin_rows": len([row for row in matched_kernel_rows if row["execution_unit_key"] == unit_key]),
                "missing_ov_kernel_entry_rows": len([row for row in missing_kernel_entry_rows if row["execution_unit_key"] == unit_key]),
                "primitive_ids": join_unique_values(row.primitive_id for row in group),
                "implementations": join_unique_values(row.implementation for row in group),
            })

        steady_ov_rows = [row for row in ov_kernel_rows if row.net_id_num != 0]
        steady_net_ids = sorted({row.net_id_num for row in steady_ov_rows})
        stable_networks: List[str] = []
        unstable_networks: List[str] = []
        for steady_net_id in steady_net_ids:
            rows_for_net = [row for row in steady_ov_rows if row.net_id_num == steady_net_id]
            counts_per_unit = sorted(
                {len([item for item in rows_for_net if item.execution_unit_key == unit_key]) for unit_key in {row.execution_unit_key for row in rows_for_net}}
            )
            if len(counts_per_unit) == 1:
                stable_networks.append(str(steady_net_id))
            else:
                unstable_networks.append(str(steady_net_id))

        steady_unit_count = len({row.execution_unit_key for row in steady_ov_rows})
        setup_unit_count = len({row.execution_unit_key for row in ov_kernel_rows if row.net_id_num == 0})
        direct_count_match = len(gtpin_kernel_rows) == len(direct_joined_rows)

        if kernel_name == MISSING_OV_KERNEL_ENTRY_LABEL:
            status = "ov_missing_kernel_entry_label"
            reason = "Synthetic bucket for OpenVINO dispatch rows whose kernel_entry was absent."
        elif missing_kernel_entry_rows:
            status = "ov_kernel_entry_missing"
            reason = "OpenVINO dispatch rows exist for these GTPin dispatches, but the OV kernel_entry was missing."
        elif gtpin_kernel_rows and not ov_kernel_rows:
            status = "gtpin_only_kernel"
            reason = "Kernel appears only in the GTPin input."
        elif ov_kernel_rows and not gtpin_kernel_rows:
            status = "ov_only_kernel"
            reason = "Kernel appears only in the OpenVINO dispatch dump."
        elif not direct_count_match:
            status = "dispatch_id_count_mismatch"
            reason = "Direct dispatch-id join count does not match the GTPin kernel count."
        elif unstable_networks:
            status = "non_uniform_within_network"
            reason = "Kernel does not repeat uniformly inside steady repeated network(s): " + ",".join(unstable_networks)
        elif setup_unit_count > 0 and steady_unit_count == 0:
            status = "setup_only_kernel"
            reason = "Kernel appears only in the internal/setup network candidate."
        else:
            status = "robust_for_multi_inference"
            reason = "Kernel matches by dispatch id and is stable within each repeated steady network."

        kernel_summary_rows.append({
            "model_label": model_label,
            "kernel_entry": kernel_name,
            "ov_dispatch_rows_total": len(ov_kernel_rows),
            "gtpin_dispatch_rows_total": len(gtpin_kernel_rows),
            "direct_matched_dispatch_rows": len(direct_joined_rows),
            "missing_ov_kernel_entry_rows": len(missing_kernel_entry_rows),
            "setup_execution_unit_count": setup_unit_count,
            "steady_execution_unit_count": steady_unit_count,
            "steady_network_ids": ";".join(steady_net_ids and [str(item) for item in steady_net_ids] or []),
            "stable_steady_network_ids": ";".join(stable_networks),
            "unstable_steady_network_ids": ";".join(unstable_networks),
            "direct_count_match": bool_text(direct_count_match),
            "partition_status": status,
            "partition_reason": reason,
            "primitive_ids": join_unique_values(row.primitive_id for row in ov_kernel_rows),
            "implementations": join_unique_values(row.implementation for row in ov_kernel_rows),
        })

    return kernel_execution_unit_rows, kernel_summary_rows


def write_report(
    path: Path,
    parse_format: str,
    joined_rows: Sequence[Dict[str, object]],
    execution_unit_rows: Sequence[Dict[str, object]],
    kernel_summary_rows: Sequence[Dict[str, object]],
) -> None:
    matched_rows = [row for row in joined_rows if row["ov_match"] == "matched"]
    missing_ov_rows = [row for row in joined_rows if row["ov_match"] == "missing_ov_dispatch"]
    missing_ov_kernel_rows = [row for row in joined_rows if row["ov_match"] == "missing_ov_kernel_entry"]
    setup_execution_units = [row for row in execution_unit_rows if row["is_internal_network_candidate"] == "true"]
    steady_execution_units = [row for row in execution_unit_rows if row["is_internal_network_candidate"] != "true"]
    robust_kernels = [row for row in kernel_summary_rows if row["partition_status"] == "robust_for_multi_inference"]
    problem_kernels = [row for row in kernel_summary_rows if row["partition_status"] != "robust_for_multi_inference"]
    setup_only_kernels = [row for row in kernel_summary_rows if row["partition_status"] == "setup_only_kernel"]
    missing_kernel_entry_kernels = [
        row for row in kernel_summary_rows
        if row["partition_status"] in {"ov_kernel_entry_missing", "ov_missing_kernel_entry_label"}
    ]
    pointer_coverage_matches = [
        row for row in joined_rows
        if row["ov_match"] != "missing_ov_dispatch" and row["pointer_coverage_match"] == "true"
    ]
    kernel_name_matches = [row for row in matched_rows if row["kernel_name_match"] == "true"]

    lines = [
        "# Multi-Inference Robustness Report",
        "",
        "## Summary",
        "",
        "- Inputs used by this script are the OpenVINO dispatch dump plus a normalized GTPin input.",
        "- Direct correlation still uses GTPin DispatchId <-> OpenVINO global_dispatch_id.",
        "- The streamlined downstream artifact is `dispatch_gtpin_kernel_metrics_join.csv`.",
        f"- Parsed GTPin input format: {parse_format}.",
        "",
        f"- GTPin dispatch rows parsed: {len(joined_rows)}",
        f"- Directly matched rows: {len(matched_rows)}",
        f"- GTPin rows missing OpenVINO partner: {len(missing_ov_rows)}",
        f"- OpenVINO rows with missing OV kernel entry: {len(missing_ov_kernel_rows)}",
        f"- Rows with pointer coverage agreement: {len(pointer_coverage_matches)}",
        f"- Rows with kernel-name agreement: {len(kernel_name_matches)}",
        f"- Setup execution units: {len(setup_execution_units)}",
        f"- Steady execution units: {len(steady_execution_units)}",
        f"- Kernels robust for multi-inference: {len(robust_kernels)}",
        f"- Kernels needing review: {len(problem_kernels)}",
        f"- Setup-only kernels: {len(setup_only_kernels)}",
        f"- Kernels affected by missing OV kernel entry labels: {len(missing_kernel_entry_kernels)}",
        "",
        "## Outputs",
        "",
        "- `dispatch_gtpin_multi_inference_join.csv`",
        "- `dispatch_gtpin_kernel_metrics_join.csv`",
        "- `dispatches_missing_in_gtpin.csv`",
        "- `execution_unit_summary.csv`",
        "- `network_summary.csv`",
        "- `kernel_execution_unit_breakdown.csv`",
        "- `kernel_multi_inference_summary.csv`",
    ]

    if problem_kernels:
        lines.extend(["", "## Example Kernel Partitions", ""])
        for row in sorted(problem_kernels, key=lambda item: str(item["kernel_entry"]))[:15]:
            lines.append(
                f"- {row['kernel_entry']}: {row['partition_status']} ({row['partition_reason']})"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dispatch_csv = Path(args.dispatch_csv)
    gtpin_dump = Path(args.gtpin_dump)
    output_dir = Path(args.output_dir) if args.output_dir else dispatch_csv.parent / "multi_inference_robustness_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dispatch_csv.exists():
        raise FileNotFoundError(f"Missing dispatch csv: {dispatch_csv}")
    if not gtpin_dump.exists():
        raise FileNotFoundError(f"Missing GTPin dump: {gtpin_dump}")

    dispatch_rows = parse_dispatch_csv(dispatch_csv)
    parse_format, tool_kind, gtpin_rows = parse_gtpin_dispatch_rows(gtpin_dump, args.gtpin_input_format, args.tool_kind)
    if not gtpin_rows:
        raise RuntimeError(f"Parsed zero GTPin dispatch rows from {gtpin_dump}")

    ov_by_global_id = {str(row.global_dispatch_id): row for row in dispatch_rows}
    gtpin_by_dispatch_id = {str(row.dispatch_id): row for row in gtpin_rows}

    joined_rows: List[Dict[str, object]] = []
    streamlined_rows: List[Dict[str, object]] = []
    for gtpin_row in gtpin_rows:
        ov_row = ov_by_global_id.get(str(gtpin_row.dispatch_id))
        joined_row = format_join_row(args.model_label, ov_row, gtpin_row)
        joined_rows.append(joined_row)
        streamlined_rows.append({field: joined_row.get(field, "") for field in STREAMLINED_JOIN_FIELDS})

    ov_only_rows: List[Dict[str, object]] = []
    for ov_row in dispatch_rows:
        if str(ov_row.global_dispatch_id) not in gtpin_by_dispatch_id:
            ov_only_rows.append({
                "model_label": args.model_label,
                "dispatch_id": ov_row.global_dispatch_id,
                "net_id": ov_row.net_id,
                "iteration": ov_row.iteration,
                "execution_unit_key": ov_row.execution_unit_key,
                "primitive_id": ov_row.primitive_id,
                "primitive_type": ov_row.primitive_type,
                "implementation": ov_row.implementation,
                "kernel_entry": ov_row.kernel_entry_effective,
                "ov_kernel_entry_present": ov_row.has_kernel_entry,
                "reason": "missing_gtpin_dispatch",
            })

    execution_unit_rows = summarize_execution_units(args.model_label, dispatch_rows, joined_rows, ov_only_rows)
    network_rows = summarize_networks(args.model_label, dispatch_rows, execution_unit_rows)
    kernel_execution_unit_rows, kernel_summary_rows = summarize_kernels(
        args.model_label,
        dispatch_rows,
        gtpin_rows,
        joined_rows,
        execution_unit_rows,
    )

    write_csv(output_dir / "dispatch_gtpin_multi_inference_join.csv", sorted(joined_rows, key=lambda item: to_int(item["dispatch_id"], default=0)), STANDARD_JOIN_FIELDS + PERFORMANCE_METRIC_FIELDS + MEMORY_METRIC_FIELDS)
    write_csv(output_dir / "dispatch_gtpin_kernel_metrics_join.csv", sorted(streamlined_rows, key=lambda item: to_int(item["dispatch_id"], default=0)), STREAMLINED_JOIN_FIELDS)
    write_csv(output_dir / "dispatches_missing_in_gtpin.csv", sorted(ov_only_rows, key=lambda item: to_int(item["dispatch_id"], default=0)), [
        "model_label", "dispatch_id", "net_id", "iteration", "execution_unit_key", "primitive_id", "primitive_type", "implementation", "kernel_entry", "ov_kernel_entry_present", "reason"
    ])
    write_csv(output_dir / "execution_unit_summary.csv", sorted(execution_unit_rows, key=lambda item: to_int(item["first_dispatch_id"], default=0)), [
        "model_label", "execution_unit_key", "net_id", "iteration", "is_internal_network_candidate", "logical_execution_index", "ov_dispatch_rows", "matched_gtpin_rows", "missing_ov_kernel_entry_rows", "missing_gtpin_rows", "distinct_ov_kernels", "distinct_matched_gtpin_kernels", "first_dispatch_id", "last_dispatch_id"
    ])
    write_csv(output_dir / "network_summary.csv", sorted(network_rows, key=lambda item: to_int(item["net_id"], default=0)), [
        "model_label", "net_id", "is_internal_network_candidate", "execution_unit_count", "rows_total", "rows_per_execution_unit", "distinct_row_counts_per_execution_unit", "distinct_kernel_entries", "first_dispatch_id", "last_dispatch_id", "interpretation"
    ])
    write_csv(output_dir / "kernel_execution_unit_breakdown.csv", sorted(kernel_execution_unit_rows, key=lambda item: (str(item["kernel_entry"]), str(item["logical_execution_index"]), to_int(item["net_id"], default=0), to_int(item["iteration"], default=0))), [
        "model_label", "kernel_entry", "execution_unit_key", "net_id", "iteration", "is_internal_network_candidate", "logical_execution_index", "ov_dispatch_rows", "matched_gtpin_rows", "missing_ov_kernel_entry_rows", "primitive_ids", "implementations"
    ])
    write_csv(output_dir / "kernel_multi_inference_summary.csv", sorted(kernel_summary_rows, key=lambda item: str(item["kernel_entry"])), [
        "model_label", "kernel_entry", "ov_dispatch_rows_total", "gtpin_dispatch_rows_total", "direct_matched_dispatch_rows", "missing_ov_kernel_entry_rows", "setup_execution_unit_count", "steady_execution_unit_count", "steady_network_ids", "stable_steady_network_ids", "unstable_steady_network_ids", "direct_count_match", "partition_status", "partition_reason", "primitive_ids", "implementations"
    ])
    write_report(output_dir / "multi_inference_robustness_report.md", parse_format, joined_rows, execution_unit_rows, kernel_summary_rows)

    print(
        json.dumps(
            {
                "parsed_format": parse_format,
                "tool_kind": tool_kind,
                "gtpin_rows": len(gtpin_rows),
                "joined_rows": len(joined_rows),
                "missing_gtpin_rows": len(ov_only_rows),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
