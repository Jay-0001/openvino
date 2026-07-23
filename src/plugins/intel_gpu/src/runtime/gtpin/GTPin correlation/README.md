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

## Notes

- `--build-info-dir` is accepted as an alias of `--build-info-root`
- `--source-dir` is accepted as an alias of `--source-root`
- the parser ignores `igc_check`
- duplicate kernel names are preserved in occurrence-level output and consolidated in identity-level output
- shared-kernel ambiguity is made explicit through `mapping_kind`, `primitive_fanout`, and `mapped_primitive_ids`
- markdown tables are emitted automatically alongside the CSV files into the same `--output-dir`
- markdown entries are separated with blank lines for readability and are optimized for narrow preview panes
