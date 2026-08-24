# Graph Join Preparation Report

## Summary

- Dispatch rows total: 5677
- Top-down rows total after normalization: 11781
- Execution units total: 12
- Dispatch rows without top-down primitive match: 155
- Top-down primitive rows without dispatch match: 6270
- Primitive nodes with more than one kernel entry: 11
- Execution units with inference-scoped top-down rows: 11
- Execution units using replicated network-scoped topology: 0

## Execution Units

- n0_i0: net_id=0, iteration=0, dispatch_rows=155, topdown_primitives=0, mapped_primitives=0, topdown_mode=missing_topdown
- n1_i0: net_id=1, iteration=0, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i1: net_id=1, iteration=1, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i10: net_id=1, iteration=10, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i2: net_id=1, iteration=2, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i3: net_id=1, iteration=3, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i4: net_id=1, iteration=4, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i5: net_id=1, iteration=5, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i6: net_id=1, iteration=6, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i7: net_id=1, iteration=7, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i8: net_id=1, iteration=8, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
- n1_i9: net_id=1, iteration=9, dispatch_rows=502, topdown_primitives=1071, mapped_primitives=501, topdown_mode=inference_scoped
