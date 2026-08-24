# GTPin Runtime Integration

## Overview

This directory contains the OpenVINO GPU runtime integration for Intel GTPin.

The code in this directory is responsible for:

* loading the GTPin runtime libraries at GPU-plugin startup
* loading an external GTPin tool selected through environment configuration
* registering that tool through `GTPin_Entry()`
* starting profiling early enough for GPU kernel instrumentation to observe inference execution

The integration dynamically loads:

* GTPin runtime libraries
* External GTPin tool DLLs
* Registers tools through `GTPin_Entry()`

before OpenCL device discovery begins.

The current orchestration point is:

```text
ocl_device_detector::create_device_list()
```

which initializes GTPin before the primary OpenCL platform and device discovery path used by the GPU runtime.

---

## Directory Contents

The current integration layer is centered around:

* `gtpin_shared_library.hpp/.cpp`
  runtime DLL loading and symbol resolution
* `gtpin_session.hpp/.cpp`
  session-level orchestration and tool registration
* `gtpin_profiler.hpp/.cpp`
  higher-level runtime control used by the GPU plugin
* `readme.md`
  integration notes for this directory
* `gsoc26_design_doc.md`
  consolidated design summary for the GSoC 2026 integration and correlation work

The companion documentation for adjacent work lives in:

* `GTPin Tools/`
  custom external tools loaded by this integration
* `GTPin correlation/`
  downstream analysis and correlation scripts

---

## Build Configuration

Enable the integration during OpenVINO configuration:

```bash
cmake ^
  -DENABLE_GTPIN_INTEGRATION=ON ^
  -DGTPIN_ROOT="<path-to-gtpin>/Profilers" ^
  ...
```

`GTPIN_ROOT` should point to the root GTPin `Profilers` directory.

Expected layout:

```text
Profilers/
|-- Include/
|-- Lib/
\-- Examples/
```

The build system automatically derives:

```text
GTPIN_INCLUDE_DIR
GTPIN_API_INCLUDE_DIR
GTPIN_GED_INCLUDE_DIR
```

from `GTPIN_ROOT`.

---

## Runtime Configuration

Before launching OpenVINO applications, configure the following environment variables.

### GTPin Runtime Location

Directory containing:

```text
ged.dll
gtpin_core.dll
iga_wrapper.dll
gtpin.dll
```

Example:

```powershell
$env:OV_GTPIN_RUNTIME_DIR="<path-to-gtpin>\Profilers\Lib\intel64"
```

---

### GTPin Tool Path

Path to the external GTPin tool DLL.

Example:

```powershell
$env:OV_GTPIN_TOOL_PATH="<path-to-gtpin>\Profilers\Examples\intel64\<tool>.dll"
```

For the custom tools maintained in this fork, the DLL path should point to a built artifact produced from the sources documented in `GTPin Tools/README.md`.

---

## Level Zero Requirements

To use GT-Pin tools with oneAPI Level Zero, the following environment variables must be enabled:

```powershell
$env:ZE_ENABLE_TRACING_LAYER="1"
$env:ZET_ENABLE_PROGRAM_INSTRUMENTATION="1"
```

These variables are required for kernel instrumentation and were observed to be essential for successful profiling collection during OpenVINO GPU inference.

---

## Example

```powershell
$env:OV_GTPIN_RUNTIME_DIR="<path-to-gtpin>\Profilers\Lib\intel64"
$env:OV_GTPIN_TOOL_PATH="<path-to-gtpin>\Profilers\Examples\intel64\<tool>.dll"
$env:ZE_ENABLE_TRACING_LAYER="1"
$env:ZET_ENABLE_PROGRAM_INSTRUMENTATION="1"

.\model_creation_sample.exe model.bin GPU
```

---

## Profiling Output

The output location is determined by the loaded GTPin tool.

For example, a loaded tool may generate:

```text
gtpin_profile/
```

or tool-specific text and TSV reports collected during OpenVINO inference.

The exact output schema depends on the selected tool.

---

## Current Status

* Dynamic GTPin runtime loading
* Dynamic external tool loading
* Tool registration through `GTPin_Entry()`
* Early initialization before OpenCL device discovery
* Profiling data collection during OpenVINO GPU inference
* Environment-variable based runtime configuration
* Verified collection of GTPin profiling data from OpenVINO GPU workloads
* Support for custom tool-based profiling flows used by this fork

---

## See Also

* [GSoC 2026 design doc](./gsoc26_design_doc.md)
* [GTPin Tools](./GTPin%20Tools/README.md)
* [GTPin correlation pipeline](./GTPin%20correlation/README.md)
