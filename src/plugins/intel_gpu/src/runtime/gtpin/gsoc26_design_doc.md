# gsoc26 design doc

---

## Overview

This document summarizes the GSoC 2026 work around Intel GTPin support inside the OpenVINO Intel GPU plugin. The project grew from basic tool loading into a layered profiling and analysis system:

1. a runtime integration layer inside the GPU plugin that can initialize GTPin early enough for kernel instrumentation,
2. a dispatch and topology artifact layer emitted by OpenVINO during execution, and
3. an offline correlation and visualization pipeline that joins GTPin outputs with OpenVINO execution structure for repeated-inference analysis.

The main design choice was to keep the runtime layer narrow and push most interpretation into downstream tools. The runtime was designed to:

- dynamically load the GTPin runtime libraries,
- dynamically load an external GTPin tool selected through environment variables,
- register the tool through `GTPin_Entry()`,
- initialize early in the OCL device creation path, and
- emit enough OpenVINO-side execution metadata to support later correlation.

This separation matters because runtime integration and analysis have different constraints. Runtime code must stay minimal and safe for inference. Correlation and graph logic need flexibility and iteration. The project therefore treats OpenVINO as the source of structured execution artifacts and GTPin as the source of kernel-level measurements, then joins them offline.

Architecturally, the work falls into four blocks:

- **Runtime bootstrap and ownership**
  Initialize GTPin before the main OpenCL execution path. This includes library loading, session creation, tool registration, and ownership.

- **Execution-side metadata capture**
  Emit dispatch metadata and top-down primitive hierarchy data in a form that supports offline joins. This includes `dispatch_map.csv`, execution-unit identifiers, kernel-entry preservation, and topology rows.

- **Offline correlation**
  Align GTPin dumps with OpenVINO dispatches and enrich them with primitive and hierarchy information. This includes dispatch-id joins, pointer validation, repeated-inference partitioning, and ambiguity handling.

- **Graph and hotspot interpretation**
  Produce per-inference hotspot bundles and navigable views across component path, op type, origin op, primitive, and kernel.

The resulting system is deliberately hybrid. The plugin performs the minimum runtime work needed to expose instrumentation and emit artifacts. The heavier logic, including robustness analysis, graph preparation, and HTML visualization, lives in the Python correlation pipeline so it can evolve without destabilizing core runtime code.

---

## Required environment variables

The runtime integration depends on a small set of explicit environment variables:

- `OV_GTPIN_RUNTIME_DIR`
  Points to the directory that contains the GTPin runtime DLLs such as `gtpin.dll`, `gtpin_core.dll`, `ged.dll`, and `iga_wrapper.dll`.

- `OV_GTPIN_TOOL_PATH`
  Points to the external GTPin tool DLL to be loaded by the plugin. This can target a stock GTPin tool or a custom tool built for this fork.

- `ZE_ENABLE_TRACING_LAYER=1`
  Enables the Level Zero tracing layer required for instrumented execution.

- `ZET_ENABLE_PROGRAM_INSTRUMENTATION=1`
  Enables Level Zero program instrumentation so kernel profiling can occur.

At build time, `GTPIN_ROOT` is also required through CMake configuration. It is not a runtime environment variable, but it is part of the integration contract because it anchors the GTPin headers, libraries, and example tool layout.

---

## Methodology

The methodology follows a staged artifact flow rather than a monolithic analysis flow.

1. **Enable runtime instrumentation**
   Build OpenVINO with GTPin integration enabled, initialize GTPin before OCL device discovery, and verify that a selected tool loads correctly.

2. **Capture OpenVINO execution artifacts**
   Run inference while dumping `dispatch_map.csv` and top-down primitive rows so OpenVINO-side execution structure is preserved.

3. **Collect GTPin tool output**
   Run one GTPin tool per analysis invocation and capture its kernel-level output in a stable text format.

4. **Perform the first correlation join**
   Use `multi_inference_robustness_analysis.py` to align GTPin rows with OpenVINO dispatch rows, partition repeated inference into execution units, and generate the streamlined downstream join.

5. **Prepare graph and hotspot bundles**
   Use `graph_join_preparation.py` to attach hierarchy information and materialize execution-unit-aware hotspot bundles.

6. **Generate drilldown views**
   Use `hotspot_table_visualization.py` to render static HTML views for hotspot-centric inspection instead of relying on raw CSV files alone.

This methodology reflects the main design philosophy of the project: keep runtime capture small, keep intermediate artifacts auditable, and allow correlation logic to evolve offline.

---

## Modified files and relevance of modifications

This section groups the major modified files by the functionality they enabled.

### 1. Core runtime integration and GTPin ownership

These files establish the basic GTPin integration inside the GPU plugin.

- `src/runtime/gtpin/gtpin_shared_library.hpp`
- `src/runtime/gtpin/gtpin_shared_library.cpp`
- `src/runtime/gtpin/gtpin_session.hpp`
- `src/runtime/gtpin/gtpin_session.cpp`
- `src/runtime/gtpin/gtpin_profiler.hpp`
- `src/runtime/gtpin/gtpin_profiler.cpp`
- `src/runtime/gtpin/readme.md`

**Relevance of these modifications**

These files define the runtime-facing GTPin layer. They handle DLL loading, symbol lookup, session lifecycle, tool registration, and the profiler control surface used by the plugin.

This group is foundational because none of the downstream correlation work matters unless GTPin can be loaded, initialized, and kept alive during inference.

---

### 2. Plugin and runtime orchestration entry points

These files are where the plugin-side orchestration was connected to the live GPU execution path.

- `src/runtime/ocl/ocl_device_detector.cpp`
- `src/runtime/CMakeLists.txt`
- `src/plugin/plugin.cpp`
- `src/plugin/graph.cpp`
- `src/plugin/compiled_model.cpp`
- `src/plugin/sync_infer_request.cpp`
- `src/runtime/engine.cpp`

**Relevance of these modifications**

This group matters because GTPin must be initialized at the correct orchestration point. `ocl_device_detector.cpp` is especially important because it became the bootstrap point for the integration before normal device discovery proceeds too far.

The surrounding plugin and engine files matter because they determine how execution state, model lifecycle, and infer-request flow behave under the integrated profiling path.

---

### 3. Build options, internal properties, and integration switches

These files expose or support the integration from the build and configuration side.

- `CMakeLists.txt`
- `include/intel_gpu/runtime/internal_properties.hpp`
- `include/intel_gpu/runtime/options.inl`
- `include/intel_gpu/graph/network.hpp`

**Relevance of these modifications**

These modifications make GTPin support discoverable and controllable at configuration time. They provide build-time enablement and internal property flow so the runtime knows when to activate integration or dump behavior.

Without this layer, the integration would remain ad hoc and harder to maintain.

---

### 4. Dispatch dump generation and topology capture in the graph runtime

These files enabled the OpenVINO-side artifacts required for offline correlation.

- `src/graph/network.cpp`
- `include/intel_gpu/graph/network.hpp`

**Relevance of these modifications**

`network.cpp` is the bridge between runtime execution and offline analysis. The work here introduced or extended:

- unified dispatch dumping,
- global dispatch identifiers,
- kernel-entry fallback handling,
- address capture for input and output arguments,
- top-down primitive map artifacts,
- iteration-aware execution-unit labeling,
- origin-op metadata capture, and
- repeated-inference-aware topology dumping.

This is the file group that turned the project from raw profiling integration into a correlation-capable system by creating the OpenVINO-side artifact contract used later by the Python pipeline.

---

### 5. Kernel identity preservation across OCL primitive implementations

These files were modified to preserve or recover kernel identity across different execution paths.

- `src/graph/impls/ocl/primitive_base.hpp`
- `src/graph/impls/ocl/multi_stage_primitive.hpp`
- `src/graph/impls/ocl/custom_primitive.cpp`
- `src/graph/impls/ocl/gemm.cpp`
- `src/graph/impls/ocl/kv_cache.cpp`
- `src/graph/impls/ocl_v2/primitive_ocl_base.hpp`
- `src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp`
- `src/graph/impls/ocl_v2/moe/moe_router_fused_opt.cpp`

**Relevance of these modifications**

This group exists because correlation quality is only as good as kernel identity quality. Not every execution path exposes a clean `kernelString->entry_point`, so fallback mechanisms were added to recover identity from runtime kernel objects where needed.

These changes prevent dispatch rows from degrading into anonymous or blank entries and ensure that multi-stage, custom, GEMM, KV-cache, and OCL v2 MOE paths participate in the same correlation flow.

---

### 6. Offline correlation pipeline

These files implement the maintained offline analysis path under the GTPin correlation directory.

- `src/runtime/gtpin/GTPin correlation/README.md`
- `src/runtime/gtpin/GTPin correlation/multi_inference_robustness_analysis.py`
- `src/runtime/gtpin/GTPin correlation/graph_join_preparation.py`
- `src/runtime/gtpin/GTPin correlation/hotspot_table_visualization.py`

**Relevance of these modifications**

This group is the current maintained analysis pipeline. It is where the project moved from "runtime dumps exist" to "runtime dumps can be interpreted."

- `multi_inference_robustness_analysis.py`
  Performs the first stable join between OpenVINO dispatch rows and GTPin rows. It establishes the dispatch-level contract, supports different GTPin input kinds, preserves generic metric fields, and partitions repeated inference into execution units.

- `graph_join_preparation.py`
  Takes the streamlined correlation output and the OpenVINO top-down dump, then materializes a hierarchy-friendly bundle for drilldown.

- `hotspot_table_visualization.py`
  Defines the current presentation layer. Instead of forcing a full DAG as the only experience, it treats hotspot ranking and grouped hierarchy views as the primary inspection path.

The directory `README.md` matters because it documents the maintained Python-first, hotspot-first end-to-end flow.

---

### 7. Documentation and design continuity

These files keep the work understandable as a system rather than as isolated code edits.

- `src/runtime/gtpin/readme.md`
- `src/runtime/gtpin/GTPin correlation/README.md`
- `src/runtime/gtpin/gsoc26_design_doc.md`

**Relevance of these modifications**

This group matters because the project spans runtime integration, artifact dumping, correlation, and visualization. Without explicit documentation, the work is easy to misread as either only a loader integration or only a set of offline scripts.

These documents explain the intended boundary between runtime logic and analysis logic, which is one of the central architectural decisions of the project.

---

## Challenges

The first challenge was orchestration. GTPin had to be initialized early enough to observe real GPU execution, but not so invasively that it destabilized normal device discovery or plugin startup.

The second challenge was runtime ownership and configuration. Because the integration is environment-driven and external-tool-driven, success depends on correct runtime DLL paths, correct tool selection, and the right Level Zero instrumentation settings.

The third challenge was kernel identity consistency. Different OCL execution paths do not expose kernel metadata uniformly, so correlation-quality dispatch dumps required fallback handling across primitive families instead of relying on a single clean entry-point source.

The fourth challenge was artifact design. The runtime had to emit enough OpenVINO-side structure to support offline reasoning, but the dump format still had to remain narrow enough to be maintainable inside core plugin code.

The fifth challenge was ambiguity in correlation. Some kernels map cleanly to primitives, but shared kernels, setup kernels, and reorder kernels do not. That forced the project toward explicit ambiguity reporting instead of naive one-to-one matching.

The sixth challenge was repeated inference. Dispatch-level execution and topology-level structure do not always arrive with the same scope or completeness, so execution-unit-aware joins and inference-scoped hierarchy preparation became necessary before visualization could be trusted.

---

## Limitations

The runtime integration works, but it is still configuration-sensitive. It depends on correct environment variables, external tool paths, and Level Zero instrumentation settings rather than offering a fully self-validating user experience.

Kernel identity handling is much stronger than in the early prototypes, but it is still partly normalized through fallback logic. That means the runtime does not yet provide one perfectly uniform kernel-identity surface across all execution paths.

The correlation pipeline is credible, but not exhaustive. Shared kernels remain the main ambiguous class, and some support kernels still resist clean one-to-one interpretation.

Repeated-inference correlation is stronger than earlier count-based approaches, but topology coverage is not uniformly complete across all execution units. Some later units can still appear with missing top-down coverage in graph-prep summaries.

The current visualization path is hotspot-first rather than topology-complete. That is intentional, but it also means the maintained user experience prioritizes ranked inspection over a fully faithful runtime DAG reconstruction.

---

## Future work

The first future direction is **online correlation**. The current system is intentionally offline and artifact-driven, but the next major step is to explore how much correlation can be surfaced during or immediately after execution without overloading the runtime path.

The second direction is **stronger ambiguous-kernel resolution**. Shared kernels, setup kernels, and reorder-related kernels still limit full end-to-end interpretability, so improving their classification and correlation confidence remains the main analytical priority.

The third direction is **cleaner execution-unit-aware graph coverage**. More inference slices should carry complete topology context so repeated-inference analysis, hotspot grouping, and structural drilldown stay consistent across the full run.
