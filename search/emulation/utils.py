"""
Utility classes: NVFP4Utils and DataGenerator
Fully aligned with verify_acc_modeling.py
"""
import torch


class NVFP4Utils:
    """负责底层映射与解包"""
    _TABLE_CACHE = {}

    @staticmethod
    def get_fp4_e2m1_table(device="cuda"):
        key = str(device)
        if key not in NVFP4Utils._TABLE_CACHE:
            pos_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            neg_vals = [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
            NVFP4Utils._TABLE_CACHE[key] = torch.tensor(pos_vals + neg_vals, device=device, dtype=torch.float16)
        return NVFP4Utils._TABLE_CACHE[key]

    @staticmethod
    def unpack_nvfp4_to_fp16(packed_uint8, original_shape):
        device = packed_uint8.device
        table = NVFP4Utils.get_fp4_e2m1_table(device)
        low = packed_uint8 & 0x0F
        high = (packed_uint8 >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).view(original_shape)
        return table[unpacked.long()]


def nvfp_unpack_fp4_to_fp16(packed_uint8: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
    """NVFP E2M1: packed ``uint8`` (2 nibbles/byte) -> ``float16`` matrix of shape ``original_shape``."""
    return NVFP4Utils.unpack_nvfp4_to_fp16(packed_uint8, original_shape)


def nvfp_swizzled_block_scale_to_linear(scale: torch.Tensor, rows: int, num_groups: int) -> torch.Tensor:
    """NVFP: swizzled FP8 block scales -> linear ``(rows, num_groups)`` ``float32``."""
    import nvfp.pseudo_quant as pseudo_quant

    return pseudo_quant.swizzled_to_linear_128_4(scale, rows, num_groups).to(torch.float32)
