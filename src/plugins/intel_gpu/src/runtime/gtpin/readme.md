# GTPin Runtime Integration

## Overview

This directory contains the OpenVINO GPU runtime integration for Intel GTPin.

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
├── Include/
├── Lib/
└── Examples/
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

For example, the `funtime` tool generates:

```text
gtpin_profile/
```

containing profiling reports collected during OpenVINO inference.

---

## Current Status

* Dynamic GTPin runtime loading
* Dynamic external tool loading
* Tool registration through `GTPin_Entry()`
* Early initialization before OpenCL device discovery
* Profiling data collection during OpenVINO GPU inference
* Environment-variable based runtime configuration
* Verified collection of GTPin profiling data from OpenVINO GPU workloads
  """
