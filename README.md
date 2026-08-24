# OpenVINO Fork Overview

This fork focuses on extending the Intel GPU plugin with Intel GTPin-based profiling and a supporting analysis workflow for connecting low-level kernel activity to OpenVINO runtime structure.

---

## Project Summary

The work in this fork is centered on three connected areas:

- GTPin runtime integration in the Intel GPU plugin
- custom external GTPin tools for collecting dispatch, execution, and memory-oriented profiling data
- downstream analysis scripts for correlating tool outputs with OpenVINO dispatch and graph metadata

The implementation lives primarily under `src/plugins/intel_gpu/src/runtime/gtpin/`.

---

## What This Fork Adds

Compared with the upstream baseline, this fork adds:

- runtime-side loading of GTPin libraries and external GTPin tools
- early GTPin initialization before GPU device discovery completes
- custom GTPin tools tailored to OpenVINO GPU inference analysis
- a documented path from runtime profiling to offline correlation analysis

---

## Main Documentation

Start with the documents below, depending on what you need:

- [Intel GPU plugin README](./src/plugins/intel_gpu/README.md)
- [Fork copy of the original top-level OpenVINO README](./ov_readme.md)
- [GTPin runtime integration](./src/plugins/intel_gpu/src/runtime/gtpin/readme.md)
- [GSoC 2026 GTPin design doc](./src/plugins/intel_gpu/src/runtime/gtpin/gsoc26_design_doc.md)
- [GTPin Tools](./src/plugins/intel_gpu/src/runtime/gtpin/GTPin%20Tools/README.md)
- [GTPin correlation pipeline](./src/plugins/intel_gpu/src/runtime/gtpin/GTPin%20correlation/README.md)

---

## High-Level Workflow

1. OpenVINO GPU runtime initializes GTPin support.
2. A selected external GTPin tool is loaded and registered.
3. GPU inference runs under GTPin instrumentation.
4. The tool emits raw profiling outputs.
5. Downstream scripts align those outputs with OpenVINO dispatch-level metadata.

---

## Directory Guide

The main directories relevant to this project are:

- `src/plugins/intel_gpu/src/runtime/gtpin/`
  runtime integration layer
- `src/plugins/intel_gpu/src/runtime/gtpin/GTPin Tools/`
  custom external profiling tools
- `src/plugins/intel_gpu/src/runtime/gtpin/GTPin correlation/`
  downstream analysis and correlation scripts

---

## Current Status

At the current stage of the fork:

- runtime integration is implemented and documented
- the custom GTPin tool set is present in-source
- correlation-side work is present but may continue evolving independently
- the README files above are the main entry points for understanding the project structure

---

## Scope Note

This README is fork-specific. The preserved upstream-style OpenVINO top-level README content is available at [ov_readme.md](./ov_readme.md).

