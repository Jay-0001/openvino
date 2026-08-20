# GTPin Offline Correlation

This folder contains a standalone offline analysis utility for correlating:

- GTPin profiling output
- `build_implementations.info`
- dumped OpenCL source buckets

The utility does not require the OpenVINO runtime environment or GTPin SDK libraries at execution time. It only consumes already-generated dump files.

---

## Build

From this directory:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

This produces:

```text
build\Release\offline_correlation.exe
```

For single-config generators, the executable may instead be:

```text
build\offline_correlation.exe
```

---

## Inputs

Required inputs:

- `--gtpin-profile <file>`
- at least one `--build-info <file>` or a directory via `--build-info-root <dir>`
- at least one `--source-bucket <file>` or a directory via `--source-root <dir>`
- `--output-dir <dir>`

---

## Default Output Mode

By default, the tool limits output to the current focused 7-key-kernel ResNet slice. This keeps the first generated table small, auditable, and aligned with the current validation strategy.

To emit the full dataset instead, add:

```powershell
--all-kernels
```

To emit a custom subset, repeat:

```powershell
--kernel <kernel_entry>
```

---

## Example Run

```powershell
.\build\Release\offline_correlation.exe `
  --gtpin-profile "W:\Building\GSOC\Before Acceptance\openvino_pre\bin\intel64\Release\GTPIN_PROFILE_17\Session_Final\resnet_single_infer.txt" `
  --build-info-root "W:\Building\GSOC\Coding_period\Correlation_analysis\dumps\resnet2\graphs" `
  --source-root "W:\Building\GSOC\Coding_period\Correlation_analysis\dumps\resnet2\sources" `
  --output-dir "W:\Building\GSOC\Coding_period\Correlation_analysis\offline_outputs\resnet2"
```

Generated files:

- `kernel_occurrences.csv`
- `kernel_identity_summary.csv`
- `kernel_occurrences.md`
- `kernel_identity_summary.md`

The markdown outputs are intentionally compact summaries rather than wide tables. They keep only the highest-information correlation fields and omit bulky items such as source bucket paths.

The CSV outputs now also expose ambiguity fields for higher-layer correlation:

- `mapping_kind`
  - `unique_primitive`
  - `shared_primitives`
  - `unmatched`
- `primitive_fanout`
- `mapped_primitive_ids`

---

## Dispatch Alignment And Lineage

The runtime-side dispatch dump now supports a unified `dispatch_map.csv` layout with:

- `global_dispatch_id`
- `input_arg_addresses`
- `output_arg_addresses`
- `output_memory_addresses`

This format is designed for direct dispatch-to-dispatch matching against GTPin `DispatchId` and for address-based producer-consumer analysis.

Use the dedicated analyzer:

```powershell
.\dispatch_lineage_correlation_analysis.ps1 `
  -DispatchCsv "W:\path\to\dispatch_map.csv" `
  -GtpinDump "W:\path\to\kernel_arg_address_dump.txt" `
  -OutputDir "W:\path\to\dispatch_lineage_analysis"
```

Generated files:

- `dispatch_gtpin_join.csv`
- `buffer_lineage_edges.csv`
- `buffer_lineage_nearest_consumers.csv`
- `dispatch_lineage_summary.md`

The older `dispatch_correlation_analysis.ps1` script is still useful for count-based kernel occurrence checks and now accepts either legacy `dispatch_map_raw*.csv` files or the unified `dispatch_map.csv`.

---

## Multi-Inference Robustness

For repeated-inference experiments, use the dedicated analyzer below instead of the lineage script. It works directly from:

- the OpenVINO `dispatch_map.csv`
- the raw GTPin kernel-argument dispatch dump such as `kernel_arg_address_dump.txt`

It first aligns dispatches through `DispatchId <-> global_dispatch_id`, then evaluates whether repeated inference behavior is stable across OpenVINO iterations.

```powershell
.\multi_inference_robustness_analysis.ps1 `
  -DispatchCsv "W:\path\to\dispatch_map.csv" `
  -GtpinDump "W:\path\to\kernel_arg_address_dump.txt" `
  -ModelLabel "mobilevnetv3_model" `
  -OutputDir "W:\path\to\multi_inference_analysis"
```

Generated files:

- `dispatch_gtpin_multi_inference_join.csv`
- `dispatches_missing_in_gtpin.csv`
- `iteration_summary.csv`
- `kernel_iteration_breakdown.csv`
- `kernel_multi_inference_summary.csv`
- `multi_inference_robustness_report.md`

This script intentionally does not build producer-consumer graphs. Its current job is narrower:

- verify that direct GTPin/OpenVINO dispatch alignment is intact
- verify that OpenVINO dispatch footprints repeat cleanly across iterations
- surface kernels that break repeated-inference assumptions under multi-inference runs

---

## Graph Preparation And Visualization

The graph-prep step materializes the current correlation hierarchy:

- `component_group -> op_type -> layer -> primitive -> kernel`

Use:

```powershell
python .\graph_join_preparation.py `
  --dispatch-join "W:\path\to\dispatch_gtpin_kernel_metrics_join.csv" `
  --topdown "W:\path\to\ov_topdown_primitive_rows.csv" `
  --output-dir "W:\path\to\graph_join_prep" `
  --bundle-name "tinyllama_graph_prep"
```

For the current primary visualization, generate Cytoscape.js DAG pages from that bundle:

```powershell
python .\graph_visualization_cytoscape.py `
  --graph-prep-root "W:\path\to\graph_join_prep" `
  --output-dir "W:\path\to\graph_join_prep\cytoscape_dag" `
  --mode filtered `
  --layout dagre
```

Notes:

- the Cytoscape DAG view keeps hierarchy edges as the main layered structure
- primitive dependency edges are included as an optional overlay, not as layout-driving edges
- older HTML-only experiments were moved under `static_html\`

---

## Notes

- `--build-info-dir` is accepted as an alias of `--build-info-root`
- `--source-dir` is accepted as an alias of `--source-root`
- the parser ignores `igc_check`
- duplicate kernel names are preserved in occurrence-level output and consolidated in identity-level output
- shared-kernel ambiguity is made explicit through `mapping_kind`, `primitive_fanout`, and `mapped_primitive_ids`
- markdown tables are emitted automatically alongside the CSV files into the same `--output-dir`
- markdown entries are separated with blank lines for readability and are optimized for narrow preview panes
