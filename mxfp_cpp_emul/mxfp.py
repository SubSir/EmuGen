import torch
import flashinfer

# E2M1: code in low 3 bits = magnitude, bit 3 = sign (same table as mxfp_emulation / dequant.cpp LUT)
_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def swizzled_sf_to_linear(
    scale_swizzled: torch.Tensor, mn: int, num_sf_columns: int
) -> torch.Tensor:
    """Undo SWIZZLED_128x4 (same reshape+transpose as mxfp_emulation::swizzled_to_linear_128_4)."""
    mn_pad, sf_k_pad = scale_swizzled.shape
    m_tiles = mn_pad // 128
    k_tiles = sf_k_pad // 4
    tmp = scale_swizzled.reshape(m_tiles, k_tiles, 32, 4, 4).transpose(1, 3).contiguous()
    linear = tmp.reshape(mn_pad, sf_k_pad)
    return linear[:mn, :num_sf_columns]


def mxfp4_scale_uint8_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    """E8M0 scale bytes -> float (matches typical __nv_fp8_e8m0 / OCP MX exponent scale)."""
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def unpack_mxfp4_indices(packed: torch.Tensor) -> torch.Tensor:
    """(M, K/2) uint8 -> (M, K) indices 0..15; low nibble first, then high nibble per byte."""
    p = packed.to(torch.uint8)
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    stacked = torch.stack((low, high), dim=-1)
    return stacked.reshape(*p.shape[:-1], p.shape[-1] * 2).to(torch.long)


def dequant_mxfp4(
    fp4_packed: torch.Tensor,
    scale_swizzled: torch.Tensor,
    *,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequant MXFP4: packed E2M1 weights + swizzled E8M0 (uint8) scales.

    fp4_packed: (M, K/2) uint8
    scale_swizzled: (pad_up(M,128), pad_up(K/group_size,4)) uint8 from mxfp4_quantize
    """
    if fp4_packed.dim() != 2:
        raise ValueError("fp4_packed must be 2D")
    M, kh = fp4_packed.shape
    K = kh * 2
    if K % group_size != 0:
        raise ValueError("K must be divisible by group_size")
    G = K // group_size

    lut = _FP4_LUT.to(device=fp4_packed.device, dtype=torch.float32)
    vals = lut[unpack_mxfp4_indices(fp4_packed)]

    s_lin = mxfp4_scale_uint8_to_float(swizzled_sf_to_linear(scale_swizzled, M, G))
    s_per_k = s_lin.unsqueeze(-1).expand(M, G, group_size).reshape(M, K).contiguous()

    return (vals * s_per_k).to(out_dtype)


if __name__ == "__main__":
    m, n, k = 512 * 4, 1024 * 4, 768 * 4
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    a_fp4, a_scale = flashinfer.mxfp4_quantize(a)
    b_fp4, b_scale = flashinfer.mxfp4_quantize(b)

    a_deq = dequant_mxfp4(a_fp4, a_scale, group_size=32, out_dtype=torch.bfloat16)
    b_deq = dequant_mxfp4(b_fp4, b_scale, group_size=32, out_dtype=torch.bfloat16)
    print("dequant max abs err vs a (row 1, cols 0:2):", (a_deq - a).abs().max().item())

    out = flashinfer.gemm.mm_fp4(
        a=a_fp4,
        b=b_fp4.T,
        a_descale=a_scale,
        b_descale=b_scale.T,
        out_dtype=torch.bfloat16,
        block_size=32,
        use_nvfp4=False,
        backend="cudnn",
    )

    print(f"Input shapes: a={a.shape}, b={b.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(out)

    print((a_deq @ b_deq.T - out).abs().max().item())
