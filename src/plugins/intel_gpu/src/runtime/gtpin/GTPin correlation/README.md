# GTPin Correlation Pipeline

This directory contains the current offline correlation pipeline used to join OpenVINO GPU dispatch dumps with GTPin outputs, prepare hotspot-first graph artifacts, and generate static HTML analysis views.

The active pipeline is Python-only:

1. `multi_inference_robustness_analysis.py`
2. `graph_join_preparation.py`
3. `hotspot_table_visualization.py`


---

## Pipeline Overview

### Stage 1: Dispatch + GTPin Join

Use `multi_inference_robustness_analysis.py` as the first join.

It consumes:

- OpenVINO `dispatch_map.csv`
- one GTPin text dump per run

It aligns rows primarily through:

- `DispatchId` from GTPin
- `global_dispatch_id` from the OpenVINO dump

It then emits both:

- a detailed correlation table
- a streamlined downstream join used by graph preparation

Supported GTPin input kinds:

- `exec_profile`
- `memory_axis`
- `raw_dispatch_dump`
- `auto` detection

Supported tool labels:

- `performance`
- `memory`
- `raw_dispatch`
- `auto`

Example:

```powershell
python .\multi_inference_robustness_analysis.py `
  --dispatch-csv "<path-to-dispatch-map>\dispatch_map.csv" `
  --gtpin-dump "<path-to-gtpin-dump>\raw_exec_kernel_correlation.txt" `
  --model-label "tinyllama_model" `
  --output-dir "<path-to-output-dir>"
```

Important outputs:

- `dispatch_gtpin_multi_inference_join.csv`
- `dispatch_gtpin_kernel_metrics_join.csv`
- `dispatches_missing_in_gtpin.csv`
- `execution_unit_summary.csv`
- `network_summary.csv`
- `kernel_execution_unit_breakdown.csv`
- `kernel_multi_inference_summary.csv`
- `multi_inference_robustness_report.md`

### Current Join Contract

The streamlined downstream artifact is:

- `dispatch_gtpin_kernel_metrics_join.csv`

This file is intentionally tool-agnostic at the correlation layer. The common contract includes:

- dispatch and OpenVINO alignment fields
- `tool_kind`
- `primary_metric_name`
- `primary_metric_total`
- `primary_metric_avg`
- `primary_metric_unit`
- `gtpin_metric_fields_json`

The JSON payload preserves tool-specific metric fields so later stages can recover extra metrics without requiring a wide fixed schema for every tool.

Today the primary metric policy is:

- performance: `gtpin_total_execution_cycles`
- memory: `gtpin_estimated_total_bytes`

The first join may also carry explicit tool-specific columns when they are useful downstream, but the generic primary-metric contract is the stable interface.

---

## Stage 2: Graph-Join Preparation

Use `graph_join_preparation.py` on the streamlined join plus the OpenVINO top-down dump.

Inputs:

- `dispatch_gtpin_kernel_metrics_join.csv`
- `ov_topdown_primitive_rows*.csv`

Example:

```powershell
python .\graph_join_preparation.py `
  --dispatch-join "<path-to-join>\dispatch_gtpin_kernel_metrics_join.csv" `
  --topdown "<path-to-topdown>\ov_topdown_primitive_rows.csv" `
  --output-dir "<path-to-output-dir>" `
  --bundle-name "tinyllama_graph_join"
```

This stage builds a hotspot-first hierarchy rather than reconstructing the full execution graph. The main structure is:

- component path
- resolved op type
- origin op
- primitive
- kernel

Primary goals:

- preserve per-inference hotspot rows
- attach higher-level OpenVINO hierarchy
- generate lightweight graph rows for later drilldown
- keep the output bundle compact and visualization-friendly

Main bundle outputs:

- `<bundle-name>_execution_units.csv`
- `<bundle-name>_hotspot_table.csv`
- `<bundle-name>_hierarchy_rows.csv`
- `<bundle-name>_graph_nodes.csv`
- `<bundle-name>_graph_edges.csv`
- `<bundle-name>_summary.json`
- `<bundle-name>_report.md`

This stage also writes one sub-bundle per execution unit with matching hotspot, hierarchy, graph, and summary files.

### Metric Handling In Graph Prep

Graph prep is still bundle-level single-tool per invocation.

That means:

- one generated bundle should represent one `tool_kind`
- multiple inferences inside that bundle are fine
- mixing performance and memory rows inside the same bundle is not supported

The script uses the generic fields:

- `primary_metric_name`
- `primary_metric_total`
- `primary_metric_avg`
- `primary_metric_unit`

It can also recover tool-specific fields from `gtpin_metric_fields_json` when needed for hotspot tables and detail views.

---

## Stage 3: Static Hotspot Visualization

Use `hotspot_table_visualization.py` on the graph-prep bundle.

Example:

```powershell
python .\hotspot_table_visualization.py `
  --graph-prep-root "<path-to-graph-prep-root>" `
  --output-dir "<path-to-html-output>"
```

This visualization is table-first and hotspot-first. It does not try to render the full runtime DAG as the primary user experience.

Current behavior:

- ranks rows by `primary_metric_total`
- uses `primary_metric_avg` or a tool-specific secondary metric where appropriate
- derives titles, labels, and units from `primary_metric_name` and `primary_metric_unit`
- preserves structural grouping views
- enforces exactly one `tool_kind` per bundle

Current grouped views:

- hierarchy
- primitive type
- kernel family

Current tool-specific behavior:

- performance bundles use cycle-oriented summaries based on the generic primary metric contract
- memory bundles use memory-oriented detail fields and composition-style views rather than pretending that run-to-run byte totals are the most informative chart

Generated outputs include:

- `index.html`
- one HTML page per execution unit

---

## Recommended End-To-End Flow

1. Generate OpenVINO `dispatch_map.csv` and top-down primitive rows.
2. Run GTPin for exactly one tool per analysis invocation.
3. Run `multi_inference_robustness_analysis.py` to produce the streamlined join.
4. Run `graph_join_preparation.py` to build the hotspot bundle.
5. Run `hotspot_table_visualization.py` to generate the HTML views.

This is the maintained path for repeated-inference hotspot analysis.

---

## Practical Notes

- `multi_inference_robustness_analysis.py` is the source of truth for the first join. The older PowerShell wrapper is no longer part of the maintained flow.
- The pipeline is designed to be metric-aware without becoming tool-fragmented too early. The first join stays generic, while later stages can still expose tool-specific details.
- The graph-prep and visualization stages assume the join has already been validated for dispatch alignment.
- The visualization bundle must contain exactly one `tool_kind`.
- Visualization logic is intentionally hotspot-centric and hierarchy-centric rather than topology-complete.

---

## Relevant Files In This Folder

- `multi_inference_robustness_analysis.py`
- `graph_join_preparation.py`
- `hotspot_table_visualization.py`
- `README.md`

These are the files that define the maintained correlation pipeline in this directory.

