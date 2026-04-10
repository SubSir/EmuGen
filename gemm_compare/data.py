"""Random tensor generators for GEMM accuracy tests."""
import torch


class DataGenerator:
    @staticmethod
    def get_random_tensor(shape, dist_type, device="cuda", dtype=torch.float16):
        if dist_type == "normal":
            return torch.randn(shape, device=device, dtype=dtype)
        if dist_type == "uniform":
            return torch.rand(shape, device=device, dtype=dtype) * 2 - 1
        if dist_type == "large":
            return torch.randn(shape, device=device, dtype=dtype) * 100.0
        if dist_type == "small":
            return torch.randn(shape, device=device, dtype=dtype) * 0.001
        if dist_type == "outliers":
            t = torch.randn(shape, device=device, dtype=dtype)
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        if dist_type == "mixed_rows":
            t = torch.randn(shape, device=device, dtype=dtype)
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        if dist_type == "abs_large":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        raise ValueError(f"Unknown distribution type: {dist_type}")


class DataGenerator_Abs:
    """Positive-only variants (handy for debugging)."""

    @staticmethod
    def get_random_tensor(shape, dist_type, device="cuda", dtype=torch.float16):
        if dist_type == "normal":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype))
        if dist_type == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)
        if dist_type == "large":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        if dist_type == "small":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 0.001
        if dist_type == "outliers":
            t = torch.abs(torch.randn(shape, device=device, dtype=dtype))
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        if dist_type == "mixed_rows":
            t = torch.abs(torch.randn(shape, device=device, dtype=dtype))
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        if dist_type == "abs_large":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        raise ValueError(f"Unknown distribution type: {dist_type}")
