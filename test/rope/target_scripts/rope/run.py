from pathlib import Path

import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent

load(
    name="qknorm_rope_ext_impl",
    sources=[
        str(root / "csrc" / "bindings.cpp"),
        str(root / "csrc" / "fused_qknorm_rope_kernel.cu"),
    ],
    extra_include_paths=[
        str(root / "csrc"),
    ],
    extra_cflags=[
        "-O3",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
    ],
    verbose=True,
    is_python_module=False,
)

op_name = "qknorm_rope_ext::fused_qk_norm_rope"

print(torch._C._dispatch_dump_table(op_name))

assert hasattr(
    torch.ops.qknorm_rope_ext,
    "fused_qk_norm_rope",
)

fused_op = torch.ops.qknorm_rope_ext.fused_qk_norm_rope.default

print(
    torch._C._dispatch_has_kernel_for_dispatch_key(
        "qknorm_rope_ext::fused_qk_norm_rope",
        "CUDA",
    )
)