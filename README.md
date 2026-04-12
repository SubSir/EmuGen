# EmuGen

Tools and experiments for floating-point GEMM emulation and accuracy comparison against hardware or library implementations.

## GEMM comparison framework (`gemm_compare/`)

The `gemm_compare` package runs randomized tests by wiring three callables:

1. **`quant_fn(a, b)`** — Takes full-precision activations/matrices and returns an opaque **state** (packed tensors, scales, metadata).
2. **`real_fn(state)`** — Hardware or fast library path (e.g. Cutlass, FlashInfer).
3. **`emul_fn(state)`** — Emulation or reference path to compare against `real_fn`.

The runner generates `A ∈ ℝ^{M×K}` and `B ∈ ℝ^{N×K}` with configurable distributions, runs both paths, and reports **bit-exact** agreement (with shared NaN handling), mismatch samples, and suite summaries.

### Layout

| Path | Purpose |
|------|---------|
| `gemm_compare/runner.py` | `compare_tensors`, `run_test_case`, `run_suite` |
| `gemm_compare/data.py` | `DataGenerator`, `DataGenerator_Abs` (random tensor distributions) |
| `gemm_compare/backends/nvfp.py` | NVFP4: `build_nvfp_fns()` |
| `gemm_compare/backends/mxfp.py` | MXFP4: `build_mxfp_fns()` |
| `gemm_compare/cli.py` | Command-line entry for NVFP / MXFP suites |
| `gemm_compare/examples/toy_cpu.py` | Minimal three-function example (CPU, no extra deps beyond PyTorch) |

### Python API

From the repository root, ensure the root is on `PYTHONPATH` (or install the tree as a package):

```python
from gemm_compare.backends.nvfp import build_nvfp_fns
from gemm_compare import run_suite

quant_fn, real_fn, emul_fn, meta = build_nvfp_fns(out_dtype=torch.float16)

exit_code = run_suite(
    quant_fn,
    real_fn,
    emul_fn,
    name="NVFP Cutlass vs C++ emulation",
    num_iterations=100,
    device="cuda",
    dtype=torch.float16,
)
# 0 = all exact matches, 1 = at least one mismatch
```

For MXFP, use `torch.bfloat16` inputs and `build_mxfp_fns()` (FlashInfer + dequant reference).

### Command line

```bash
# NVFP: nvfp.ops (Cutlass) vs nvfp_cpp_emul (JIT C++ extension)
PYTHONPATH=. python -m gemm_compare nvfp -n 50

# MXFP: flashinfer.gemm.mm_fp4 vs dequantized BF16 matmul (see mxfp_cpp_emul/mxfp.py)
PYTHONPATH=. python -m gemm_compare mxfp -n 20 --mxfp-backend cudnn --group-size 32
```

Options include `--seed`, `--iterations` / `-n`, NVFP `--nvfp-out float16|bfloat16`, and MXFP `--mxfp-backend`, `--group-size`.

### Backends

**NVFP (`gemm_compare/backends/nvfp.py`)**  
- **Quant:** `nvfp.pseudo_quant.pytorch_nvfp4_quantize` with the same global-scale recipe as the original verifier.  
- **Real:** `nvfp.ops.cutlass_scaled_fp4_mm`.  
- **Emul:** `nvfp_cpp_emul.emulated_scaled_fp4_mm` (extension under `nvfp_cpp_emul/`; parent directory is added to `sys.path` automatically).

**MXFP (`gemm_compare/backends/mxfp.py`)**  
- **Quant:** `flashinfer.mxfp4_quantize`.  
- **Real:** `flashinfer.gemm.mm_fp4` with `use_nvfp4=False` (OCP-style MX path).  
- **Emul:** `dequant_mxfp4` from `mxfp_cpp_emul/mxfp.py`, then `A_deq @ B_deq.T` as a reference.

### Related files

- **`mxfp_cpp_emul/mxfp.py`** — MXFP4 dequantization helpers and an optional demo guarded by `if __name__ == "__main__"` so imports stay side-effect free.  
- **`nvfp_cpp_emul/`** — JIT-built libtorch extension and scripts (`example.py`, `speed_bench_hw_vs_emul.py`) for NVFP emulation vs hardware.  
- **`test.py` / `utils.py`** — Legacy verifier snippets from another repo (depend on `ops`, `emulation` there). The supported NVFP workflow in *this* repo is the `gemm_compare` NVFP backend plus `nvfp_cpp_emul`.

### Requirements (by backend)

- **Toy example:** PyTorch only.  
- **NVFP backend:** CUDA, `nvfp` stack, and a successful JIT compile of `nvfp_cpp_emul`.  
- **MXFP backend:** CUDA, `flashinfer`, and compatible `mm_fp4` backend (e.g. cuDNN).
