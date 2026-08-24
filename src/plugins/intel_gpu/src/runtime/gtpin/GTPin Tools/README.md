# GTPin Tools

This directory stores the custom external GTPin tools used by this fork.

---

## Purpose

These tools collect profiling data that is not emitted by the OpenVINO runtime alone. Their outputs feed the correlation workflow documented in the neighboring correlation directory.

Each tool is built as an external GTPin tool and then loaded by the OpenVINO GPU plugin through the runtime integration layer in `../readme.md`.

---

## Current Tools

- `kernel_arg_address_dump.cpp`
  Captures dispatch-level kernel argument address information.

- `raw_exec_kernel_correlation.cpp`
- `raw_exec_kernel_correlation.h`
  Captures execution-oriented dispatch data together with memory-argument context.

- `memory_axis_kernel_correlation.cpp`
- `memory_axis_kernel_correlation.h`
  Captures memory-oriented kernel activity and dispatch context.

The current tool set is source-complete at the repository level:

* a dispatch-address dump tool
* a raw execution correlation tool
* a memory-axis correlation tool

No extra generated outputs or compiled binaries are intended to live in this directory.

---

## Practical Roles

Use the tools at a high level as follows:

* `kernel_arg_address_dump`
  best suited when the downstream analysis needs argument addresses and dispatch alignment
* `raw_exec_kernel_correlation`
  best suited when execution-oriented kernel summaries and dispatch-level argument context are needed together
* `memory_axis_kernel_correlation`
  best suited when memory-heavy behavior is the main profiling target

---

## Expected Outputs

The exact files depend on the selected tool and knobs, but the tool family currently targets:

* readable text summaries
* per-dispatch structured outputs
* per-argument or per-kernel structured outputs where applicable

These outputs are intended to be consumed by the downstream analysis flow rather than committed back into the source tree.

---

## Relationship To The Pipeline

At a high level:

1. a GTPin tool is loaded through the OpenVINO GTPin runtime integration
2. the tool emits raw profiling outputs
3. the correlation scripts consume those outputs for alignment and analysis

For the integration layer, see [../readme.md](../readme.md).

For the correlation workflow, see [../GTPin correlation/README.md](../GTPin%20correlation/README.md).

---

## Maintenance Notes

To keep this directory submission-ready:

* keep only source and documentation files here
* avoid committing built DLLs or temporary profiling outputs
* update this README when tool responsibilities or output formats change materially
