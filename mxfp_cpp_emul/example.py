#!/usr/bin/env python3
"""
Example: MXFP4 GEMM via FlashInfer (same pipeline as sibling ``mxfp.py`` in this folder).

Run as ``python mxfp_cpp_emul/example.py`` from the repo root. Requires ``flashinfer``
and a CUDA device. MXFP4 uses ``block_size=32`` and ``use_nvfp4=False``; only the
``cudnn`` backend supports MXFP4 (``cutlass`` / ``trtllm`` expect NVFP4 with
``block_size=16``).

Optional JIT compile flags for the sibling C++ emulation package (not used here)::

  import nvfp_cpp_emul
  nvfp_cpp_emul.JIT_EXTRA_CFLAGS.append("-O3")

For NVFP4 scaled GEMM + libtorch emulation comparison, see ``nvfp_cpp_emul/example.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Directory that contains ``nvfp_cpp_emul/`` (JIT extension sources).
_EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))


def main() -> None:
    import torch
    import flashinfer

    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    m, n, k = 512, 1024, 768
    torch.manual_seed(0)

    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.zeros(n, k, dtype=torch.bfloat16, device=device)
    b[:, 0] = 1.0

    # Quantize to MXFP4 format
    a_fp4, a_scale = flashinfer.mxfp4_quantize(a)
    b_fp4, b_scale = flashinfer.mxfp4_quantize(b)

    # Perform MXFP4 GEMM (b transposed for column-major layout)
    out = flashinfer.gemm.mm_fp4(
        a=a_fp4,
        b=b_fp4.T,
        a_descale=a_scale,
        b_descale=b_scale.T,
        out_dtype=torch.bfloat16,
        block_size=32,  # MXFP4 uses block_size=32
        use_nvfp4=False,  # MXFP4 (not NVFP4)
        # FlashInfer: only "cudnn" supports MXFP4. "cutlass"/"trtllm" require NVFP4 (block_size=16).
        # On SM120, MXFP4+cudnn also needs cuDNN backend >= 9.14.0.
        backend="cudnn",
    )

    print(f"Input shapes: a={a.shape}, b={b.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(out)


if __name__ == "__main__":
    main()
