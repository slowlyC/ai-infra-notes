"""
低内存 Dropout
==================

在本教程中，你将编写一个内存高效的 dropout 实现，其状态仅由一个 int32 种子组成。
这与更传统的 dropout 实现不同，传统实现的状态通常由一个与输入形状相同的位掩码tensor组成。

在此过程中，你将学习到:

* PyTorch 的朴素 Dropout 实现的局限性。

* Triton 中的并行伪随机数生成。

"""

# %%
# 基线实现
# --------
#
# *dropout* 算子 [SRIVASTAVA2014]_ 是一种正则化方法:
# 输入向量中的每个元素以概率 :math:`p` 置零, 否则保留。
# 这迫使网络在仅 :math:`1 - p` 的元素可用时也能正常工作。
#
# 评估时令 :math:`p=0`, 但这会增大输出范数 (可能影响 softmax 温度等)。
# 因此保留的元素乘以 :math:`\frac{1}{1 - p}` (inverted dropout),
# 使得无论 :math:`p` 取何值, 输出的期望范数保持一致。
#
# 先看基线实现:

import tabulate
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    x_ptr,  # 输入指针
    x_keep_ptr,  # 0 和 1 的掩码指针
    output_ptr,  # 输出指针
    n_elements,  # `x` tensor中的元素数量
    p,  # `x` 中的元素被改为零的概率
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # 下面这一行是关键部分，在上面的段落中描述！
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # 写回输出
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# 输入tensor
x = torch.randn(size=(10, ), device=DEVICE)
# Dropout 掩码
p = 0.5
# tensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 0], device='cuda:0', dtype=torch.int32)
x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

# %%
# 带种子的 Dropout
# --------------
#
# 上面的实现可以工作, 但有两个问题: (1) 需要存储完整的 dropout 掩码用于反向传播;
# (2) 与重计算/检查点结合时, 随机状态管理很棘手
# (参见 https://pytorch.org/docs/stable/checkpoint.html 中 `preserve_rng_state` 的讨论)。
# 下面介绍一种替代方案, 优势在于:
# (1) 内存占用更小; (2) 数据移动更少; (3) 多次调用间的随机性管理更简单。
#
# Triton 中的伪随机数生成: :code:`triton.language.rand` 接受种子和 :code:`int32` 偏移量块,
# 返回 [0, 1) 均匀分布的 :code:`float32` 值块。Triton 也提供其他
# :ref:`随机数生成策略<Random Number Generation>`。
#
# .. note::
#    Triton 的 PRNG 实现基于 Philox 算法（在 [SALMON2011]_ 中描述）。
#
# 完整实现:


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算此实例处理的元素的内存偏移量
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 从 x 加载数据
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 随机修剪它
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # 写回
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# 与基线相比 - dropout 掩码从未实例化！
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

# %%
# 只要种子相同, 就会产生完全相同的 dropout 掩码——适合重计算场景。
# 更多 GPU 随机数生成的用法可参考 `python/triton/language/random.py`。

# %%
# 练习
# ---------
#
# 1. 扩展kernel以在矩阵上操作，并使用种子向量 - 每行一个种子。
# 2. 添加对跨步的支持。
# 3. （挑战）实现一个稀疏 Johnson-Lindenstrauss 变换的kernel，每次使用种子动态生成投影矩阵。

# %%
# 参考文献
# ----------
#
# .. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
# .. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014

