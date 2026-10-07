#include <torch/extension.h>

void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    torch::Tensor& position_ids,
    int64_t forced_token_heads_per_warp);

TORCH_LIBRARY(qknorm_rope_ext, m) {
  m.def(
      "fused_qk_norm_rope("
      "Tensor(a!) qkv, "
      "int num_heads_q, "
      "int num_heads_k, "
      "int num_heads_v, "
      "int head_dim, "
      "float eps, "
      "Tensor q_weight, "
      "Tensor k_weight, "
      "Tensor cos_sin_cache, "
      "bool is_neox, "
      "Tensor position_ids, "
      "int forced_token_heads_per_warp=-1"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(qknorm_rope_ext, CUDA, m) {
  m.impl(
      "fused_qk_norm_rope",
      TORCH_FN(fused_qk_norm_rope));
}
