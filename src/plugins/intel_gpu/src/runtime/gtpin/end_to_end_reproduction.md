# Reproducing the End-to-End Workflow

This workflow reproduces the maintained GTPin profiling, offline correlation, and hotspot-visualization path. It assumes Windows, a compatible Intel GPU and driver stack, OpenVINO 2026.1.0, Intel GTPin 4.7.1, CMake, Python, and a GPU-compatible OpenVINO workload.

Run one GTPin tool per profiling invocation. Execution-oriented and memory-oriented results must be collected, correlated, and visualized as separate bundles.

## 1. Configure and build OpenVINO

Clone the project branch and configure the build from the OpenVINO repository root. `GTPIN_ROOT` must point to the Intel GTPin `Profilers` directory containing `Include`, `Lib`, and `Examples`.

```text
git clone https://github.com/Jay-0001/openvino.git
cd openvino
git checkout gsoc26-gtpin-full

cmake -S . -B build -DENABLE_GTPIN_INTEGRATION=ON -DENABLE_DEBUG_CAPS=ON -DENABLE_GPU_DEBUG_CAPS=ON -DGTPIN_ROOT="<path-to-gtpin>\Profilers"
cmake --build build --config Release
```

`ENABLE_GTPIN_INTEGRATION` adds the GTPin runtime integration to the Intel GPU plugin. `ENABLE_DEBUG_CAPS` and `ENABLE_GPU_DEBUG_CAPS` are required because the OpenVINO dispatch and top-down CSV artifacts are emitted through the GPU debug configuration.

The selected GTPin tool must be built as an external tool DLL before profiling. The custom sources in this fork are under:

```text
src\plugins\intel_gpu\src\runtime\gtpin\GTPin Tools\
```

Use `raw_exec_kernel_correlation` for execution-oriented analysis or `memory_axis_kernel_correlation` for memory-oriented analysis.

## 2. Configure the GTPin runtime

Create one output root for each model and GTPin tool run. Keep the dispatch dump, top-down rows, and GTPin output from the same run together.

```text
<output-root>\
  dispatch\
  topdown\
  gtpin\
  join\
  graph\
  html\
```

Set the following environment variables before launching the OpenVINO workload:

```ini
OV_GTPIN_RUNTIME_DIR=<path-to-gtpin>\Profilers\Lib\intel64
OV_GTPIN_TOOL_PATH=<path-to-built-tool>\raw_exec_kernel_correlation.dll
ZE_ENABLE_TRACING_LAYER=1
ZET_ENABLE_PROGRAM_INSTRUMENTATION=1
```

`OV_GTPIN_RUNTIME_DIR` must contain the required GTPin runtime DLLs, including `gtpin.dll`, `gtpin_core.dll`, `ged.dll`, and `iga_wrapper.dll`.

`OV_GTPIN_TOOL_PATH` points to one external GTPin tool DLL. The runtime loads one tool per process.

`ZE_ENABLE_TRACING_LAYER=1` and `ZET_ENABLE_PROGRAM_INSTRUMENTATION=1` are required for GTPin to observe instrumented GPU execution.

For a memory-oriented run, set `OV_GTPIN_TOOL_PATH` to:

```text
<path-to-built-tool>\memory_axis_kernel_correlation.dll
```

Use a separate output root for each tool kind.

## 3. Configure OpenVINO dispatch and top-down outputs

The correlation flow requires these OpenVINO artifacts from the same workload run:

```text
dispatch_map.csv
ov_topdown_primitive_rows<net_id>.csv
```

`GPU_DUMP_DISPATCH_MAP_PATH` and `GPU_DUMP_TOPOLOGY_PRIMITIVE_MAP_PATH` are Intel GPU configuration properties. They are not GTPin environment variables. The workload used for correlation must pass them to the GPU plugin before compiling the model:

```cpp
core.set_property("GPU", {
    {"GPU_DUMP_DISPATCH_MAP_PATH", "<output-root>/dispatch"},
    {"GPU_DUMP_TOPOLOGY_PRIMITIVE_MAP_PATH", "<output-root>/topdown"},
});
```

These properties produce:

```text
<output-root>\dispatch\dispatch_map.csv
<output-root>\topdown\ov_topdown_primitive_rows<net_id>.csv
```

The included `model_creation_sample.exe` can verify that GTPin loads and observes GPU execution, but it does not expose these GPU properties as command-line arguments. Use a workload that sets the properties above, or adapt the sample before using it for a complete correlation run.

## 4. Run an OpenVINO GPU workload

After configuring the GTPin environment and OpenVINO artifact paths, run the selected workload on `GPU`.

The included model-creation sample can be used as a GTPin-loading smoke test:

```text
model_creation_sample.exe <path-to-lenet-weights>\lenet.bin GPU
```

For repeated-inference correlation, run the intended workload for the required number of inferences. Let the process exit normally because the current GTPin tools write their final reports when the profiled process ends.

For `raw_exec_kernel_correlation`, the default readable GTPin output is:

```text
raw_exec_kernel_correlation.txt
```

Before continuing, verify that the OpenVINO and GTPin outputs from the same run are available:

```text
<output-root>\dispatch\dispatch_map.csv
<output-root>\topdown\ov_topdown_primitive_rows<net_id>.csv
<output-root>\gtpin\raw_exec_kernel_correlation.txt
```

## 5. Perform the dispatch-level join

Run the first maintained correlation stage using the OpenVINO dispatch map and one GTPin text dump.

```text
python multi_inference_robustness_analysis.py \
  --dispatch-csv "<output-root>\dispatch\dispatch_map.csv" \
  --gtpin-dump "<output-root>\gtpin\raw_exec_kernel_correlation.txt" \
  --model-label "<model-label>" \
  --output-dir "<output-root>\join"
```

Run this command from:

```text
src\plugins\intel_gpu\src\runtime\gtpin\GTPin correlation\
```

The main downstream artifact is:

```text
dispatch_gtpin_kernel_metrics_join.csv
```

Review these diagnostic outputs before continuing:

```text
dispatches_missing_in_gtpin.csv
multi_inference_robustness_report.md
```

They identify missing or ambiguous dispatch alignment instead of silently carrying it into later stages.

## 6. Attach the OpenVINO hierarchy

Use the streamlined join and the matching indexed top-down CSV to create an execution-unit-aware hotspot bundle.

```text
python graph_join_preparation.py \
  --dispatch-join "<output-root>\join\dispatch_gtpin_kernel_metrics_join.csv" \
  --topdown "<output-root>\topdown\ov_topdown_primitive_rows1.csv" \
  --output-dir "<output-root>\graph" \
  --bundle-name "<model-label>_graph_join"
```

Replace `1` in `ov_topdown_primitive_rows1.csv` with the emitted `net_id` if required.

This stage generates hotspot, hierarchy, graph, summary, and report files for the complete run and for every recovered execution unit.

## 7. Generate the HTML visualization

Generate the static HTML bundle from the graph-preparation output.

```text
python hotspot_table_visualization.py \
  --graph-prep-root "<output-root>\graph" \
  --output-dir "<output-root>\html"
```

Open `<output-root>\html\index.html` to inspect the generated bundle.

A valid result contains aligned dispatch rows, kernel and primitive metadata, one consistent `tool_kind`, and hotspot groups ranked by the selected tool's primary metric.

Do not mix execution and memory results in one correlation or visualization bundle. Repeat the workflow with the other GTPin tool to create a separate bundle.

## Expected Artifact Flow

```text
OpenVINO GPU workload
  |- dispatch_map.csv
  |- ov_topdown_primitive_rows<net_id>.csv
  `- GTPin tool output
       |
       v
multi_inference_robustness_analysis.py
  `- dispatch_gtpin_kernel_metrics_join.csv
       |
       v
graph_join_preparation.py
  `- hotspot and hierarchy bundle
       |
       v
hotspot_table_visualization.py
  `- static HTML hotspot analysis
```