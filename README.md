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
| `gemm_compare/runner.py` | `compare_tensors`, `run_test_case`, `run_suite`, `run_suite_export`, `run_suite_import` |
| `gemm_compare/data.py` | `DataGenerator`, `DataGenerator_Abs` (random tensor distributions) |
| `gemm_compare/rollout.py` | Save/load rollout `.pt` (quantized state + real output per case) |
| `gemm_compare/hf_hub_rollout.py` | Optional upload/download of rollout files on the Hugging Face Hub |
| `gemm_compare/backends/nvfp.py` | NVFP4: `build_nvfp_fns()`, `build_nvfp_emul_fn()`, state (de)serialization for rollout |
| `gemm_compare/backends/mxfp.py` | MXFP4: `build_mxfp_fns()`, `build_mxfp_emul_fn()`, state (de)serialization for rollout |
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

# NVFP pseudo vs real: pseudo_quant matmul vs Cutlass, summary metric only
PYTHONPATH=. python -m gemm_compare nvfp -n 50 --nvfp-compare pseudo-real

# MXFP: flashinfer.gemm.mm_fp4 vs dequantized BF16 matmul (see mxfp_cpp_emul/mxfp.py)
PYTHONPATH=. python -m gemm_compare mxfp -n 20 --mxfp-backend cudnn --group-size 32

# MXFP pseudo vs real: dequantized matmul vs flashinfer mm_fp4, summary metric only
PYTHONPATH=. python -m gemm_compare mxfp -n 20 --mxfp-backend cudnn --group-size 32 --mxfp-compare pseudo-real
```

Options include `--seed`, `--iterations` / `-n`, NVFP `--nvfp-out float16|bfloat16` and `--nvfp-compare real-emul|pseudo-real`, plus MXFP `--mxfp-backend`, `--group-size`, and `--mxfp-compare real-emul|pseudo-real`.

### Cross-machine rollout and Hugging Face Hub

By default the CLI uses **`--mode local`**: each case quantizes on this machine, runs **real** and **emul**, and compares them in one process.

For **split hardware** (e.g. real GEMM on one GPU box, emulation on another), use rollout artifacts:

| `--mode` | Role |
|----------|------|
| **`local`** | Same as before: `quant_fn` → `real_fn` and `emul_fn` on this machine. |
| **`export`** | On the machine with the real stack: same random suite as `local`, but each case saves **quantized state** (CPU tensors) and the **real output** into a local `.pt` file (`--artifact`). Emulation is skipped unless you pass **`--export-verify-local`** (slower sanity check). |
| **`import`** | On another machine: load that `.pt`, run **emulation only**, and compare against the **saved real outputs** bit-for-bit. |

**Export** always needs a local **`--artifact`** path (the file is written there). **Import** needs either **`--artifact`** (local file) or **`--hf-repo`** (download from the Hub; see below).

**Import-side dependencies (lighter than export):**

- **MXFP:** `build_mxfp_emul_fn()` — no FlashInfer; still needs `mxfp_cpp_emul`, `search.emulation`, and CUDA in typical setups.
- **NVFP:** `build_nvfp_emul_fn()` — no `nvfp.ops` / Cutlass; still needs `nvfp_cpp_emul` and CUDA for the C++ emulation path.

**Hugging Face Hub (optional):** install `huggingface_hub`, then you can push after export or pull on import without copying files by hand.

| Flag | Meaning |
|------|---------|
| `--hf-repo REPO_ID` | Hub repo id, e.g. `username/gemm-rollouts`. With **export**: upload after the local file is saved. With **import**: download this repo’s file (omit `--artifact`). |
| `--hf-path` | Path inside the repo (default: `gemm_compare_rollout.pt`). |
| `--hf-repo-type` | `model` (default) or `dataset`. |
| `--hf-revision` | Branch, tag, or commit. |
| `--hf-token` | API token; if omitted, `HF_TOKEN` or the cached `huggingface-cli` login is used. |
| `--hf-create-repo` | With export + `--hf-repo`: create the repo if it does not exist. |
| `--hf-private` | With `--hf-create-repo`: create a **private** repo. |

Examples:

```bash
# Machine A: write rollout locally, then push to the Hub
PYTHONPATH=. python -m gemm_compare mxfp --mode export --artifact ./mxfp_rollout.pt -n 1000 --seed 1 \
  --hf-repo YOUR_USER/gemm-rollouts --hf-path rollouts/mxfp_n1000.pt --hf-create-repo

# Machine B: import from the Hub (no local --artifact)
PYTHONPATH=. python -m gemm_compare mxfp --mode import \
  --hf-repo YOUR_USER/gemm-rollouts --hf-path rollouts/mxfp_n1000.pt

# Machine B: import from the Hub and compare pseudo vs saved real output
PYTHONPATH=. python -m gemm_compare mxfp --mode import \
  --hf-repo YOUR_USER/gemm-rollouts --hf-path rollouts/mxfp_n1000.pt \
  --import-compare pseudo

# NVFP pseudo import also works, but export must include inputs:
# add --export-include-inputs on Machine A export, then run:
# PYTHONPATH=. python -m gemm_compare nvfp --mode import \
#   --hf-repo YOUR_USER/gemm-rollouts --hf-path rollouts/nvfp_with_inputs.pt \
#   --import-compare pseudo
```

If both `--artifact` and `--hf-repo` are given for **import**, the **local `--artifact`** is used.

`pseudo-real` compare mode is for `--mode local` runs (it compares pseudo vs real directly on one machine).

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
- **Rollout Hub push/pull:** `pip install huggingface_hub` and a token or login with write access to the target repo when uploading.
