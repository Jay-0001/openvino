# OpenVINO Fork Overview

This fork focuses on extending the Intel GPU plugin with Intel GTPin-based profiling and a correlation workflow for connecting low-level kernel activity to OpenVINO runtime structure.

---

## Project Focus

The work in this fork is centered on:

- GTPin runtime integration in the Intel GPU plugin
- custom external GTPin tools for profiling dispatch and memory behavior
- a correlation pipeline for aligning GTPin outputs with OpenVINO dispatch and graph metadata

---

## Main Documentation

Start here for the plugin-level and GTPin-specific details:

- [Intel GPU plugin README](./src/plugins/intel_gpu/README.md)
- [Fork copy of the original top-level OpenVINO README](./src/plugins/intel_gpu/ov_readme.md)
- [GTPin runtime integration](./src/plugins/intel_gpu/src/runtime/gtpin/readme.md)
- [GTPin correlation pipeline](./src/plugins/intel_gpu/src/runtime/gtpin/GTPin%20correlation/README.md)
- [GTPin Tools](./src/plugins/intel_gpu/src/runtime/gtpin/GTPin%20Tools/README.md)

---

## High-Level Flow

1. OpenVINO GPU runtime initializes GTPin support.
2. A custom GTPin tool collects profiling data during inference.
3. Correlation scripts align tool outputs with OpenVINO dispatch metadata.
4. The resulting artifacts support hotspot inspection and repeated-inference analysis.

---

## Scope Note

This README is fork-specific. The preserved upstream-style OpenVINO top-level README content is available at `src/plugins/intel_gpu/ov_readme.md`.
