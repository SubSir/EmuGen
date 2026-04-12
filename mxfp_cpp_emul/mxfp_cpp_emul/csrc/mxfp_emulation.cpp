// MXFP4 scaled GEMM accuracy emulation using libtorch (matches search.emulation.core MMAEngine +
// mxfp.py: E8M0 scales, SWIZZLED_128x4, group_size 32). A-side scales promoted to fp64 before
// combine (same as Python group_size==32 path). No NVFP alpha in this entrypoint.
#include <torch/extension.h>
#include <vector>

namespace {

torch::Tensor make_fp4_e2m1_table(const torch::Device& device) {
  std::vector<float> vals = {
      0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  return torch::tensor(
      vals, torch::TensorOptions().dtype(torch::kFloat16).device(device));
}

torch::Tensor swizzled_to_linear_128_4(const torch::Tensor& a_sf, int64_t mn, int64_t num_sf_columns) {
  auto sizes = a_sf.sizes();
  TORCH_CHECK(sizes.size() == 2, "scale tensor must be 2D");
  int64_t mn_padded = sizes[0];
  int64_t sf_k_padded = sizes[1];
  int64_t m_tiles = mn_padded / 128;
  int64_t k_tiles = sf_k_padded / 4;
  TORCH_CHECK(mn_padded % 128 == 0 && sf_k_padded % 4 == 0,
              "swizzled scale shape must be divisible by 128 and 4");
  auto tmp = a_sf.reshape({m_tiles, k_tiles, 32, 4, 4}).transpose(1, 3);
  auto out = tmp.reshape({mn_padded, sf_k_padded});
  return out.narrow(0, 0, mn).narrow(1, 0, num_sf_columns);
}

torch::Tensor mxfp4_scale_uint8_to_float(const torch::Tensor& scale_u8) {
  return torch::exp2(scale_u8.to(torch::kFloat32) - 127.f);
}

torch::Tensor unpack_mxfp4_to_fp16(const torch::Tensor& packed, c10::IntArrayRef original_shape) {
  auto table = make_fp4_e2m1_table(packed.device());
  auto p = packed.to(torch::kUInt8);
  auto low = torch::bitwise_and(p, 0x0F);
  auto high = torch::bitwise_and(torch::bitwise_right_shift(p, 4), 0x0F);
  std::vector<torch::Tensor> halves = {low, high};
  auto stacked = torch::stack(halves, /*dim=*/-1);
  auto unpacked_idx = stacked.reshape(original_shape).to(torch::kLong).reshape(-1);
  auto selected = table.index_select(0, unpacked_idx).reshape(original_shape);
  return selected;
}

torch::Tensor to_float32_with_rounding(const torch::Tensor& tensor_f64, int rounding) {
  if (rounding == 0) {
    auto f32 = tensor_f64.to(torch::kFloat32);
    auto mask = f32.abs() > tensor_f64.abs();
    auto z = torch::zeros_like(f32);
    return torch::where(mask, torch::nextafter(f32, z), f32);
  }
  return tensor_f64.to(torch::kFloat32);
}

torch::Tensor hardware_reduction_nto1(
    const std::vector<torch::Tensor>& v_list,
    bool output_fp32,
    int rounding) {
  torch::Tensor v_aligned_sum =
      torch::zeros_like(v_list[0], torch::dtype(torch::kFloat64));
  for (const auto& v : v_list) {
    v_aligned_sum = v_aligned_sum + v.to(torch::kFloat64);
  }
  auto summed_f64 = v_aligned_sum;
  if (output_fp32) {
    return to_float32_with_rounding(summed_f64, rounding);
  }
  return summed_f64;
}

torch::Tensor hardware_add_wbits(
    const torch::Tensor& acc_fp32,
    const torch::Tensor& new_val_wbits,
    int rounding) {
  auto sum_f64 = acc_fp32.to(torch::kFloat64) + new_val_wbits.to(torch::kFloat64);
  return to_float32_with_rounding(sum_f64, rounding);
}

// ``k_tile`` is the K extent per stage-1 dot: 16 for NVFP-style paths, ``group_size`` (32) for MXFP
// (must match ``MMAEngine.stage1_inner_mma_fp16(..., group_size=...)`` in search/emulation/core.py).
torch::Tensor stage1_inner_mma_fp16(
    const torch::Tensor& val_a, const torch::Tensor& val_b, int64_t k_tile) {
  int64_t M = val_a.size(0);
  int64_t K = val_a.size(1);
  int64_t N = val_b.size(0);
  TORCH_CHECK(K % k_tile == 0, "K must be divisible by k_tile");
  int64_t k_groups = K / k_tile;
  auto a_grouped = val_a.view({M, k_groups, k_tile}).to(torch::kFloat16);
  auto b_grouped = val_b.view({N, k_groups, k_tile}).to(torch::kFloat16);
  auto partial = torch::einsum("mgk,ngk->mgn", {a_grouped, b_grouped}).to(torch::kFloat16);
  return partial.permute({0, 2, 1});
}

}  // namespace

torch::Tensor emulated_mxfp4_mm(
    const torch::Tensor& a_fp4,
    const torch::Tensor& b_fp4,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    int64_t group_size,
    int64_t m_chunk_size,
    int stage3_rounding,
    int stage4_rounding) {
  TORCH_CHECK(a_fp4.dim() == 2 && b_fp4.dim() == 2, "a_fp4, b_fp4 must be 2D");
  TORCH_CHECK(group_size == 32, "only MXFP4 group_size=32 is supported");
  int64_t M = a_fp4.size(0);
  int64_t N = b_fp4.size(0);
  int64_t K = a_fp4.size(1) * 2;
  TORCH_CHECK(b_fp4.size(1) * 2 == K, "K mismatch between a and b");
  TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
  int64_t G32 = K / group_size;
  constexpr int64_t k_w_reduce_mxfp = 2;
  TORCH_CHECK(
      G32 % k_w_reduce_mxfp == 0,
      "MXFP emulation (w_reduce=2) requires K/(2*group_size) to be integral (e.g. K divisible by 64)");

  auto val_b_fp16 = unpack_mxfp4_to_fp16(b_fp4, {N, K});
  auto s_b_u8 = swizzled_to_linear_128_4(scale_b, N, G32);
  auto s_a_u8 = swizzled_to_linear_128_4(scale_a, M, G32);
  auto s_b = mxfp4_scale_uint8_to_float(s_b_u8);
  auto s_a_all = mxfp4_scale_uint8_to_float(s_a_u8);
  auto val_a_fp16_all = unpack_mxfp4_to_fp16(a_fp4, {M, K});

  int64_t num_blocks = G32 / k_w_reduce_mxfp;

  std::vector<torch::Tensor> chunks;
  for (int64_t m_start = 0; m_start < M; m_start += m_chunk_size) {
    int64_t m_end = std::min(m_start + m_chunk_size, M);
    int64_t curr_m = m_end - m_start;
    auto val_a_chunk = val_a_fp16_all.narrow(0, m_start, curr_m);
    auto s_a_chunk = s_a_all.narrow(0, m_start, curr_m);

    auto ps1 = stage1_inner_mma_fp16(val_a_chunk, val_b_fp16, group_size);
    // Match MMAEngine.emulation_scaled_fp4_mm: for group_size==32, A scales in fp64 before combine.
    auto s_a_f64 = s_a_chunk.to(torch::kFloat64);
    auto combined = s_a_f64.unsqueeze(1) * s_b.unsqueeze(0);
    auto scaled = ps1.to(torch::kFloat32).to(torch::kFloat64) * combined;

    auto scaled_grouped = scaled.view({curr_m, N, num_blocks, k_w_reduce_mxfp});
    std::vector<torch::Tensor> v_pair = {
        scaled_grouped.select(-1, 0),
        scaled_grouped.select(-1, 1)};

    torch::Tensor summed_chunk;
    if (num_blocks == 1) {
      summed_chunk = hardware_reduction_nto1(
                         v_pair, /*output_fp32=*/true, stage3_rounding)
                       .squeeze(-1);
    } else {
      auto summed_groups = hardware_reduction_nto1(
          v_pair, /*output_fp32=*/false, stage3_rounding);
      auto acc = to_float32_with_rounding(summed_groups.select(-1, 0), stage4_rounding);
      for (int64_t b = 1; b < num_blocks; ++b) {
        acc = hardware_add_wbits(acc, summed_groups.select(-1, b), stage4_rounding);
      }
      summed_chunk = acc;
    }
    chunks.push_back(summed_chunk);
  }

  auto summed = torch::cat(chunks, 0);
  return summed.to(torch::kFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "emulated_mxfp4_mm",
      &emulated_mxfp4_mm,
      "MXFP4 scaled GEMM emulation (libtorch; matches search.emulation.core MMAEngine MXFP path)",
      pybind11::arg("a_fp4"),
      pybind11::arg("b_fp4"),
      pybind11::arg("scale_a"),
      pybind11::arg("scale_b"),
      pybind11::arg("group_size") = 32,
      pybind11::arg("m_chunk_size") = 128,
      pybind11::arg("stage3_rounding") = 0,
      pybind11::arg("stage4_rounding") = 0);
}
