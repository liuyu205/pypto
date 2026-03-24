# Requirement: Propagate Python Variable Names to Generated C++ Code

## Problem

Currently, generated `kernel.cpp` uses auto-incremented names (`v38`, `v39`, ...) for tile variables, making it impossible to trace back to the Python source code.

Example mapping (test_fa.py → kernel.cpp):

| C++ Name | Python Name | Tile Type |
|----------|-------------|-----------|
| `v38` | `q_mat` | Mat [128,64] FP16 |
| `v39` | `k_mat` | Mat [64,128] FP16 |
| `v46` | `qk_acc` | Acc [128,128] FP32 |
| `v48` | `qk_vec` | Vec [64,128] FP32 |
| `v51` | `reduce_dst` | Vec [64,1] CM FP32 |
| ... | ... | ... |

## Current Naming Pipeline

```
Python:     q_mat = plm.make_tile(...)
   ↓
AST Parser: ast.Name.id = "q_mat"
   ↓
IR Var:     Var("q_mat", TileType, span)   ← name preserved in var->name_
   ↓
Codegen:    var_to_mlir_["q_mat"] = "%0"   ← mapping exists, but generates pure number
   ↓
MLIR:       %0 = pto.alloc_tile ...        ← name lost
   ↓
ptoas:      v38                            ← auto-incremented
   ↓
kernel.cpp: Tile<...> v38;                 ← untraceable to q_mat
```

## Root Cause

The break point is `PTOCodegen::NewTemp()` in `src/codegen/pto/pto_codegen.cpp:677`:

```cpp
std::string PTOCodegen::NewTemp() {
  std::string name = "%" + std::to_string(temp_counter_++);  // numbers only
  last_assigned_temp_ = name;
  return name;
}
```

The Python variable name is already available via `current_result_var_name_` (set at line 564 in `VisitStmt_`) and stored in `var_to_mlir_` map, but `NewTemp()` ignores it.

## Proposed Change

### Phase 1: MLIR Name Hints (pypto codegen side)

MLIR natively supports SSA name hints (e.g., `%q_mat = ...`).

1. Modify `NewTemp()` to accept an optional name hint:

```cpp
std::string PTOCodegen::NewTemp(const std::string& hint = "") {
  std::string name = hint.empty()
      ? "%" + std::to_string(temp_counter_++)
      : "%" + hint;
  last_assigned_temp_ = name;
  return name;
}
```

2. In `VisitStmt_(AssignStmt)`, pass `current_result_var_name_` to `NewTemp()`.

3. In backend op handlers (e.g., `backend_910b_pto_ops.cpp`), pass the name hint when calling `NewTemp()`:

```cpp
std::string result = codegen.NewTemp(codegen.GetCurrentResultVarName());
```

**Result**: `kernel.pto` (MLIR) becomes readable:

```mlir
// Before
%0 = pto.alloc_tile ...
%1 = pto.alloc_tile ...

// After
%q_mat = pto.alloc_tile ...
%k_mat = pto.alloc_tile ...
```

### Phase 2: C++ Variable Names (ptoas side)

The `ptoas` compiler (external, under `/data/g00895580/Ascend/`) maps MLIR SSA values to C++ variable names. It currently uses `v{N}` numbering.

- Need to verify if `ptoas` preserves MLIR SSA name hints when generating C++.
- If not, `ptoas` would need a change to map `%q_mat` → `q_mat` instead of `v38`.

## Key Files

| File | Role |
|------|------|
| `python/pypto/language/parser/ast_parser.py` | Captures Python var name from AST (`ast.Name.id`) |
| `include/pypto/ir/expr.h:224-268` | `Var::name_` stores the Python name |
| `src/codegen/pto/pto_codegen.cpp:551-597` | `VisitStmt_` sets `current_result_var_name_` and maps to MLIR |
| `src/codegen/pto/pto_codegen.cpp:677-681` | `NewTemp()` — the break point |
| `src/backend/910B_PTO/backend_910b_pto_ops.cpp` | Backend op handlers calling `NewTemp()` |

## Impact

- Phase 1 alone makes `kernel.pto` (MLIR) debuggable — no external dependency.
- Phase 2 requires coordination with ptoas but delivers the full benefit: readable `kernel.cpp`.
