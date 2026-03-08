"""
Libdevice (`tl.extra.libdevice`) 函数
==============================
Triton 可以从外部库调用自定义函数。
本示例使用 `libdevice` 库对 tensor 应用 `asin` 函数。

参考文档:
- `CUDA libdevice-users-guide <https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_
- `HIP device-lib source code <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src>`_

`libdevice.py` 将计算相同但数据类型不同的函数聚合在一起。
例如 `__nv_asin` (double) 和 `__nv_asinf` (float) 都计算反正弦,
Triton 根据输入输出类型自动选择正确的底层函数。
"""

# %%
#  asin kernel
# ------------

import torch

import triton
import triton.language as tl
import inspect
import os
from triton.language.extra import libdevice

from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# %%
#  使用默认的 libdevice 库路径
# -----------------------------------------
# 使用 `triton/language/math.py` 中编码的默认 libdevice 库路径:

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE)
output_torch = torch.asin(x)
assert x.is_cuda and output_triton.is_cuda
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'torch 和 triton 之间的最大差异为 '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


# %%
#  自定义 libdevice 库路径
# -------------------------------------
# 也可以显式指定 libdevice 库路径:
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


current_file = inspect.getfile(inspect.currentframe())
current_dir = Path(os.path.dirname(os.path.abspath(current_file)))

if is_cuda():
    libdir = current_dir.parent.parent / 'third_party/nvidia/backend/lib'
    extern_libs = {'libdevice': str(libdir / 'libdevice.10.bc')}
elif is_hip():
    libdir = current_dir.parent.parent / 'third_party/amd/backend/lib'
    extern_libs = {}
    libs = ["ocml", "ockl"]
    for lib in libs:
        extern_libs[lib] = str(libdir / f'{lib}.bc')
else:
    raise RuntimeError('未知后端')

output_triton = torch.empty_like(x)
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
print(output_torch)
print(output_triton)
print(f'torch 和 triton 之间的最大差异为 '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

