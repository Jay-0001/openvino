# GTPin Tools

This directory stores the custom external GTPin tools used by this fork.

---

## Purpose

These tools collect profiling data that is not emitted by the OpenVINO runtime alone. Their outputs feed the correlation workflow documented in the neighboring correlation directory.

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

---

## Relationship To The Pipeline

At a high level:

1. a GTPin tool is loaded through the OpenVINO GTPin runtime integration
2. the tool emits raw profiling outputs
3. the correlation scripts consume those outputs for alignment and analysis

For the integration layer, see [../readme.md](../readme.md).

For the correlation workflow, see [../GTPin correlation/README.md](../GTPin%20correlation/README.md).
