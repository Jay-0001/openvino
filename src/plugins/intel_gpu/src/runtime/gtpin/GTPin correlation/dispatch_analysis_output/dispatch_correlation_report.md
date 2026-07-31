# Dispatch Correlation Report

## Verdict

- Total kernels assessed: 51
- Kernels sufficient for order-proxy correlation: 26
- Kernels insufficient under current artifacts: 25
- Shared-kernel entries sufficient for order-proxy correlation: 0
- Shared-kernel entries still insufficient: 12

Current conclusion: the present files are sufficient to form candidate occurrence-to-primitive mappings for the kernels whose dispatch-row counts match the offline occurrence counts. They are not sufficient as a complete end-to-end correlation solution because some support kernels, especially weight-reorder kernels, are undercounted by the network-level dump.

## Dispatch Files

- dispatch_map_raw0.csv: net_id=0, rows=108, rows_with_kernel_entry=54
- dispatch_map_raw1.csv: net_id=1, rows=60, rows_with_kernel_entry=59
- dispatch_map_raw2.csv: net_id=2, rows=60, rows_with_kernel_entry=59

## Why Local Dispatch Indices Still Work Here

- exec_index is file-local and resets per network dump, so it is not a global dispatch id.
- For this analyzer, candidate mappings use the tuple (net_id, iteration, exec_index) rather than exec_index alone.
- That is good enough for per-network ordering checks, but it remains a proxy rather than a true global launch sequence.

## Insufficient Cases

- `convolution_gpu_bfyx_f16_13720713998678496926_0_0`: expected=4, dispatch_rows=4, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_18016549827154746994_0_0`: expected=6, dispatch_rows=6, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_13143176461841059678_0_0`: expected=6, dispatch_rows=6, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_14502731020992035906_0_0`: expected=6, dispatch_rows=6, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_16212556866347025335_0_0`: expected=6, dispatch_rows=6, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_4138751954898280671_0_0`: expected=4, dispatch_rows=4, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0`: expected=12, dispatch_rows=12, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0`: expected=8, dispatch_rows=8, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_1x1_8638458454580344552_0_0`: expected=10, dispatch_rows=10, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_4065784441082281624_0_0`: expected=4, dispatch_rows=4, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_5655598150652728675_0_0`: expected=6, dispatch_rows=6, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.
- `convolution_gpu_bfyx_f16_7543907540957867454_0_0`: expected=10, dispatch_rows=10, mapping=shared_primitives
  Reason: Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection.

## Generated Files

- `dispatch_sufficiency_summary.csv`
- `dispatch_occurrence_candidates.csv`
