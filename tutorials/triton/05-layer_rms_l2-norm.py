"""
LayerNorm / RMSNorm / L2Norm
================================
在本教程中，我们实现三种常见归一化方法 LayerNorm, RMSNorm 和 L2Norm 的高性能 Triton kernel，并与纯 PyTorch 实现进行对比。

在此过程中，你将学习到:

* 三种归一化的原理与区别。

* 在 Triton 中实现反向传播。

* 在 Triton 中实现并行归约。

"""

import torch

import triton
import triton.language as tl

try:
    # 这是 https://github.com/NVIDIA/apex，不是 PyPi 上的 apex，因此不应将其添加到 setup.py 中的 extras_require。
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# %%
# ========================
# 第一部分: 三种归一化的原理与 PyTorch 实现
# ========================
#
# 本教程涵盖三种归一化方法，它们的关系如下:
#
#   LayerNorm  ──(去掉均值中心化和偏置)──▶  RMSNorm  ──(去掉权重, sum替代mean)──▶  L2Norm
#     最复杂                                  中等                                    最简单
#
# 对比:
#   ┌───────────┬──────────────────────────────────────┬──────────────┬───────────────┐
#   │ 方法       │ 公式                                 │ 归一化因子    │ 可学习参数    │
#   ├───────────┼──────────────────────────────────────┼──────────────┼───────────────┤
#   │ LayerNorm │ y = w * (x - mean) / √(var+ε) + b   │ 1/√(var+ε)   │ w, b          │
#   │ RMSNorm   │ y = w * x / √(mean(x²)+ε)           │ 1/√(mean+ε)  │ w             │
#   │ L2Norm    │ y = x / √(sum(x²)+ε)                │ 1/√(sum+ε)   │ 无            │
#   └───────────┴──────────────────────────────────────┴──────────────┴───────────────┘
#
# 参数说明:
#
# ε (epsilon) — 数值稳定性常数
#   如果输入 x 全为 0，归一化因子 = 0，导致除以 0！
#   加一个很小的 ε 防止除 0，典型值 ε = 1e-5 (LayerNorm) 或 1e-6 (RMSNorm/L2Norm)
#   例: RMS(x) = √(mean(x²) + ε)，当 x=0 时 RMS = √ε ≈ 0.001，避免了除零
#
# γ (gamma / weight) — 可学习的缩放参数
#   形状为 [N] (即每个特征维度有自己的 γ 值)，初始化为全 1
#   作用: 归一化后数值范围被压缩，γ 让模型学习"恢复"可能丢失的表达能力
#   例:
#     x = [2.0, 4.0, 6.0]          # 原始向量
#     RMS = √((4+16+36)/3) ≈ 4.32  # 均方根
#     x/RMS = [0.46, 0.93, 1.39]   # 归一化后，数值范围变小
#     γ = [1.5, 1.0, 0.8]          # 可学习参数
#     output = γ × (x/RMS) = [0.69, 0.93, 1.11]  # 恢复到合适范围
#
# β (bias / 偏置) — 可学习的平移参数 (仅 LayerNorm 有)
#   形状为 [N]，初始化为全 0
#   作用: 在缩放之后再加一个偏移，进一步增强表达能力
#   RMSNorm 去掉了 β，实践中影响不大但减少了参数量


# %%
# 1. LayerNorm (层归一化)
# --------------------------
#
# *LayerNorm* 最初在 [BA2016]_ 中引入，用于提高序列模型（如 Transformers）的训练稳定性。
# 它接受一个向量 :math:`x` 作为输入，先减去均值、除以标准差，再应用可学习的仿射变换。
#
# 前向传播:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# 其中:
#   - ε: 防止除零的小常数 (典型值 1e-5)
#   - w (gamma γ): 可学习的缩放参数，shape [N]，初始化为 1
#   - b (beta β): 可学习的平移参数，shape [N]，初始化为 0
#
# 计算流程:
#   输入向量 x = [x₁, x₂, x₃, ..., xₙ]
#                ↓
#           计算均值 μ = Σxᵢ/N
#                ↓
#           计算方差 σ² = Σ(xᵢ-μ)²/N
#                ↓
#           归一化: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)   ← ε 在这里防止除零
#                ↓
#           线性变换: yᵢ = wᵢ·x̂ᵢ + bᵢ            ← γ 缩放 + β 平移
#                ↓
#   输出向量 y = [y₁, y₂, y₃, ..., yₙ]
#
# 反向传播:
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w
#      - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x}
#      - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{and} \quad \nabla_{b} = \nabla_{y}


def pytorch_layer_norm(x, weight, bias, eps=1e-5):
    """
    纯 PyTorch 手动实现 LayerNorm (用于教学对比)

    公式: y = w * (x - mean) / √(var + ε) + b

    Args:
        x: 输入 tensor, shape (..., N)
        weight: 权重 γ, shape (N,)
        bias: 偏置 β, shape (N,)
        eps: 防止除零的小常数
    """
    # 转为 float32 计算以保证数值精度
    x_fp32 = x.float()
    # 1. 均值 μ
    mean = x_fp32.mean(dim=-1, keepdim=True)
    # 2. 方差 σ²
    var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    # 3. 归一化
    rstd = torch.rsqrt(var + eps)
    x_hat = (x_fp32 - mean) * rstd
    # 4. 仿射变换
    y = (weight.float() * x_hat + bias.float()).to(x.dtype)
    return y


# %%
# 2. RMSNorm (均方根层归一化)
# ----------------------------
#
# RMSNorm 由 Zhang & Sennrich (2019) [ZS2019]_ 提出，是 LayerNorm 的简化版本。
# 去掉了均值中心化步骤，只保留基于均方根 (RMS) 的缩放操作。
# 在 LLaMA、Gemma、Qwen、DeepSeek 等现代大语言模型中广泛使用。
#
# 前向传播:
#
# .. math::
#    \text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon}
#
#    y = w \cdot \frac{x}{\text{RMS}(x)} = w \cdot x \cdot \text{rstd}
#
# 其中:
#   - ε: 防止除零的小常数 (典型值 1e-6，比 LayerNorm 的 1e-5 更小)
#   - w (gamma γ): 可学习的缩放参数，shape [N]，初始化为 1
#     作用: 归一化将数值压缩到 [-1, 1] 附近，γ 让模型恢复需要的尺度
#   - 没有 bias β (对比 LayerNorm)，实践证明去掉 β 对效果影响很小
#
# 与 LayerNorm 的区别:
#   ┌─────────────┬──────────────────────┬──────────────────────┐
#   │             │ LayerNorm            │ RMSNorm              │
#   ├─────────────┼──────────────────────┼──────────────────────┤
#   │ 均值中心化  │ ✓ (x - mean)         │ ✗                    │
#   │ 缩放因子    │ 1/√(var + ε)         │ 1/√(mean(x²) + ε)   │
#   │ 可学习参数  │ weight + bias        │ 仅 weight            │
#   │ 前向遍历    │ 3次(mean,var,norm)   │ 2次(mean_sq,norm)    │
#   │ 应用场景    │ BERT, GPT-2, T5      │ LLaMA, Gemma, Qwen   │
#   └─────────────┴──────────────────────┴──────────────────────┘
#
# 反向传播:
#   设 x̂ = x * rstd, wdy = w ⊙ dy
#
# .. math::
#    c_1 = \frac{1}{N} \sum_i \hat{x}_i \cdot \text{wdy}_i
#
#    \nabla_x = \text{rstd} \cdot (\text{wdy} - \hat{x} \cdot c_1)
#
#    \nabla_w = \sum_{\text{rows}} dy \odot \hat{x}


def pytorch_rms_norm(x, weight, eps=1e-6):
    """
    纯 PyTorch 实现 RMSNorm

    公式: y = w * x / √(mean(x²) + ε)

    对比 LayerNorm:
      - LayerNorm: (x - mean) / √(var + ε) * w + b   ← 有 mean、有 bias
      - RMSNorm:   x / √(mean(x²) + ε) * w           ← 无 mean、无 bias
    """
    x_fp32 = x.float()
    # 1. mean(x²) — 注意是 mean 不是 sum
    mean_sq = (x_fp32 ** 2).mean(dim=-1, keepdim=True)
    # 2. rstd = 1/RMS = 1/√(mean(x²) + ε)
    rstd = torch.rsqrt(mean_sq + eps)
    # 3. y = w * x * rstd (无 bias)
    y = (weight.float() * x_fp32 * rstd).to(x.dtype)
    return y


# %%
# 3. L2Norm (L2 归一化)
# -----------------------
#
# L2Norm 将向量归一化到单位 L2 范数，将每个向量映射到单位超球面上。
# 它是最简单的归一化方式，没有可学习参数。
# 常用于特征匹配、向量检索、对比学习，也用于 linear attention 中的 QK 归一化。
#
# 前向传播:
#
# .. math::
#    y = \frac{x}{\sqrt{\sum_{i=1}^{N} x_i^2 + \epsilon}} = x \cdot \text{rstd}
#
# 其中:
#   - ε: 防止除零的小常数 (典型值 1e-6)
#   - 没有任何可学习参数 (无 γ，无 β)
#   - 输出向量的 L2 范数 ≈ 1 (被归一化到单位超球面上)
#
# L2Norm 与 RMSNorm 的关系:
#   - L2Norm:  rstd = 1/√(sum(x²) + ε)     → 除以 L2 范数
#   - RMSNorm: rstd = 1/√(sum(x²)/N + ε)   → 除以 RMS (= L2范数/√N)
#   - 区别仅在于分母是否除以 N，以及 RMSNorm 有权重 w
#
# 反向传播推导:
#   设 y = x * rstd, 其中 rstd = 1/√(sum(x²) + ε), s = sum(x²) + ε
#
#   ∂y_j/∂x_i = δ_ij * rstd + x_j * ∂rstd/∂x_i
#             = δ_ij * rstd - x_i * x_j * rstd³
#
#   ∂L/∂x_i = Σ_j dy_j * (δ_ij * rstd - x_i * x_j * rstd³)
#            = dy_i * rstd - x_i * rstd³ * Σ_j(dy_j * x_j)
#            = rstd * (dy_i - y_i * Σ_j(dy_j * y_j))
#
# .. math::
#    \nabla_x = \text{rstd} \cdot (dy - y \cdot \sum_j dy_j \cdot y_j)


def pytorch_l2_norm(x, eps=1e-6):
    """
    纯 PyTorch 实现 L2Norm

    公式: y = x / √(sum(x²) + ε)

    对比:
      - LayerNorm: (x - mean) / √(var + ε) * w + b  ← 有 mean、var、w、b
      - RMSNorm:   x / √(mean(x²) + ε) * w          ← 有 mean(x²)、w
      - L2Norm:    x / √(sum(x²) + ε)                ← 最简单，无参数
    """
    x_fp32 = x.float()
    # sum(x²) — 注意: 是 sum 不是 mean (这是与 RMSNorm 的关键区别)
    sq_sum = (x_fp32 ** 2).sum(dim=-1, keepdim=True)
    # rstd = 1/√(sum(x²) + ε)
    rstd = torch.rsqrt(sq_sum + eps)
    # y = x * rstd (无权重)
    y = (x_fp32 * rstd).to(x.dtype)
    return y


# %%
# ========================
# 第二部分: LayerNorm Triton Kernel 实现
# ========================
#
# 前向传播
# ---------
# 每个 kernel 实例处理输入矩阵的一行，需要 3 次遍历数据:
#   遍历1: 累加求和 → 计算 mean
#   遍历2: 累加平方差 → 计算 var → 得到 rstd
#   遍历3: 归一化并应用仿射变换 → 输出 y


@triton.jit
def _layer_norm_fwd_fused(
    X,  # 输入指针
    Y,  # 输出指针
    W,  # 权重指针
    B,  # 偏置指针
    Mean,  # 均值指针
    Rstd,  # 1/std 指针
    stride,  # 移动 1 行时指针增加多少
    N,  # X 中的列数
    eps,  # epsilon 以避免除以零
    BLOCK_SIZE: tl.constexpr,
):
    # 将程序 id 映射到它应该计算的 X 和 Y 的行。
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # 计算均值
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # 计算方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # 写入均值 / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # 归一化并应用线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # 写入输出
        tl.store(Y + cols, y, mask=mask)


# %%
# 反向传播
# -------------
#
# 层归一化算子的反向传播比前向传播稍微复杂一些。
# 由于同一批次中的所有行都使用相同的权重 :math:`w` 和偏置 :math:`b`，因此它们的梯度需要求和。
# 为了有效地执行此步骤，我们使用并行归约策略: 每个kernel实例将部分 :math:`\nabla_{w}` 和
# :math:`\nabla_{b}` 累积到 :math:`\text{GROUP_SIZE_M}` 个独立缓冲区之一中的某些行中。
# 这些缓冲区保留在 L2 缓存中，然后由另一个函数进一步归约以计算实际的 :math:`\nabla_{w}` 和 :math:`\nabla_{b}`。
#
#   .. image:: parallel_reduction.png
#
# 阶段 1 由函数 :code:`_layer_norm_bwd_dx_fused` 实现，阶段 2 由函数 :code:`_layer_norm_bwd_dwdb` 实现。


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # 输入梯度指针
                             DY,  # 输出梯度指针
                             DW,  # 权重梯度部分和指针
                             DB,  # 偏置梯度部分和指针
                             X,  # 输入指针
                             W,  # 权重指针
                             Mean,  # 均值指针
                             Rstd,  # 1/std 指针
                             Lock,  # 锁指针
                             stride,  # 移动 1 行时指针增加多少
                             N,  # X 中的列数
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # 将程序 id 映射到它应该计算的 X、DX 和 DY 的元素。
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # 为并行归约偏移锁和权重/偏置梯度指针
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # 将数据加载到 SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # 计算 dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # 写入 dx
    tl.store(DX + cols, dx, mask=mask)
    # 累积 dw/db 的部分和
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # 第一次存储不累积
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # 需要一个屏障来确保所有线程在释放锁之前完成
    tl.debug_barrier()

    # 释放锁
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # 权重梯度部分和指针
                         DB,  # 偏置梯度部分和指针
                         FINAL_DW,  # 权重梯度指针
                         FINAL_DB,  # 偏置梯度指针
                         M,  # GROUP_SIZE_M
                         N,  # 列数
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # 将程序 id 映射到它应该计算的 DW 和 DB 的元素。
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 遍历 DW 和 DB 的行以求和部分和。
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # 将最终和写入输出。
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# ---- LayerNorm autograd 包装 ----

class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # 分配输出
        y = torch.empty_like(x)
        # 将输入数据重塑为 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # 每个特征小于 64KB: 排队融合kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("此层归一化不支持特征维度 >= 64KB。")
        # warp 数量的启发式
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # 排队kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # 用于 DW/DB 的并行归约流数量的启发式
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # 分配输出
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # 使用前向传播启发式排队kernel
        # 同时计算 DW 和 DB 的部分和
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, _db, x, w, m, v, locks,  #
            x_arg.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        # 在单独的kernel中累积部分和
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # 创建数据
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # 前向传播
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # 反向传播（triton）
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # 反向传播（torch）
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # 比较
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


def test_pytorch_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    """验证 PyTorch 手动实现与 torch.nn.functional.layer_norm 的一致性"""
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    w = torch.rand((N,), dtype=dtype, device=device)
    b = torch.rand((N,), dtype=dtype, device=device)

    y_manual = pytorch_layer_norm(x, w, b, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps)

    max_diff = (y_manual - y_ref).abs().max().item()
    print(f"  手动实现 vs F.layer_norm: max_diff = {max_diff:.2e}", end="")
    assert torch.allclose(y_manual, y_ref, atol=1e-2, rtol=0)
    print(" ✓")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # 创建数据
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # 前向传播
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # 反向传播
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# ========================
# 第三部分: RMSNorm Triton Kernel 实现
# ========================
#
# 前向 kernel 只需 2 次遍历 (对比 LayerNorm 的 3 次):
#   遍历1: 累加 x² → 计算 mean(x²) → 得到 rstd
#   遍历2: 归一化并乘以权重 → 输出 y
#
# 反向 kernel 使用与 LayerNorm 相同的并行归约策略累积 dw，但:
#   - dx 公式少一个 c₂ 项 (因为无 mean 中心化)
#   - 无需计算 db (因为无 bias)


@triton.jit
def _rms_norm_fwd_fused(
    X,      # 输入指针
    Y,      # 输出指针
    W,      # 权重指针 (gamma)
    Rstd,   # 1/RMS 输出指针
    stride, # 行步长
    N,      # 列数 (特征维度)
    eps,    # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm 前向 kernel — 每个程序实例处理一行

    对比 LayerNorm 前向 (_layer_norm_fwd_fused):
      - LayerNorm: 需要 3 次遍历 (计算mean → 计算var → 归一化)
      - RMSNorm:   只需 2 次遍历 (计算mean(x²) → 归一化)
      - 省去了 mean 计算和 (x - mean) 的步骤
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # ---- 第一遍: 计算 sum(x²) ----
    # 对比 LayerNorm: LayerNorm 需要两遍 (先算 mean，再算 var)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    # rstd = 1/√(mean(x²) + ε)
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # ---- 第二遍: 归一化 y = w * x * rstd ----
    # 对比 LayerNorm: y = w * (x - mean) * rstd + b
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd  # 注意: 没有 (x - mean) 步骤
        y = x_hat * w      # 注意: 没有 + b 步骤
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    DX, DY, DW,         # 梯度指针
    X, W, Rstd, Lock,   # 前向保存的张量
    stride, N,           # 维度信息
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    RMSNorm 反向 kernel — 计算 dx 并累积 dw 的部分和

    对比 LayerNorm 反向 (_layer_norm_bwd_dx_fused):
      - LayerNorm: dx = rstd * (wdy - x̂*c₁ - c₂), c₂ = (1/N)Σwdy
      - RMSNorm:   dx = rstd * (wdy - x̂*c₁)  ← 无 c₂ 项 (因为无 mean 中心化)
      - LayerNorm 需要累积 dw 和 db
      - RMSNorm 只需累积 dw (无 bias)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # 并行归约偏移
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    # 加载数据
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)
    # 计算 dx:
    #   x̂ = x * rstd
    #   wdy = w ⊙ dy
    #   c₁ = (1/N) Σ(x̂ ⊙ wdy)
    #   dx = rstd * (wdy - x̂ * c₁)
    xhat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * c1) * rstd
    # 对比 LayerNorm: dx = (wdy - xhat*c1 - c2) * rstd, 其中 c2 = sum(wdy)/N
    tl.store(DX + cols, dx, mask=mask)
    # 累积 dw 部分和 (使用原子锁 — 与 LayerNorm 相同的并行归约策略)
    partial_dw = (dy * xhat).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _rms_norm_bwd_dw(
    DW, FINAL_DW,   # 部分和 → 最终梯度
    M, N,            # 维度
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """RMSNorm 反向 — 归约 dw 部分和 (阶段2，与 LayerNorm 的 _layer_norm_bwd_dwdb 类似但只有 dw)"""
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


# ---- RMSNorm autograd 包装 ----

class TritonRMSNorm(torch.autograd.Function):
    """Triton RMSNorm 的 autograd 包装 (结构与上方 LayerNorm 类一致)"""

    @staticmethod
    def forward(ctx, x, weight, eps):
        # 分配输出
        y = torch.empty_like(x)
        # 重塑为 2D
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        # BLOCK_SIZE 和 num_warps 启发式 (与 LayerNorm 相同)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("此 RMSNorm 不支持特征维度 >= 64KB。")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # 启动 kernel
        _rms_norm_fwd_fused[(M,)](
            x_arg, y, weight, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, v = ctx.saved_tensors
        N = w.shape[0]
        # GROUP_SIZE_M 启发式 (与 LayerNorm 相同)
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # 分配输出 (对比 LayerNorm: 这里没有 _db 和 db)
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # 启动 kernel
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _rms_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, x, w, v, locks,
            x_arg.stride(0), N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        _rms_norm_bwd_dw[grid](
            _dw, dw, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)
        return dx, dw, None


triton_rms_norm = TritonRMSNorm.apply


def test_rms_norm(M, N, dtype, eps=1e-6, device=DEVICE):
    """测试 RMSNorm: Triton 实现 vs PyTorch 参考实现 (前向 + 反向)"""
    # 创建数据 (注意: 先创建 x 再设 requires_grad，使其为叶子张量)
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    weight = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # Triton 前向 + 反向
    y_tri = triton_rms_norm(x, weight, eps)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri = x.grad.clone(), weight.grad.clone()
    x.grad, weight.grad = None, None
    # PyTorch 参考前向 + 反向
    y_ref = pytorch_rms_norm(x, weight, eps)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = x.grad.clone(), weight.grad.clone()
    # 比较
    fwd_diff = (y_tri - y_ref).abs().max().item()
    dx_diff = (dx_tri - dx_ref).abs().max().item()
    dw_diff = (dw_tri - dw_ref).abs().max().item()
    print(f"  前向 max_diff = {fwd_diff:.2e}", end="")
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    print(" ✓")
    print(f"  dx   max_diff = {dx_diff:.2e}", end="")
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    print(" ✓")
    print(f"  dw   max_diff = {dw_diff:.2e}", end="")
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
    print(" ✓")


# %%
# ========================
# 第四部分: L2Norm Triton Kernel 实现
# ========================
#
# L2Norm 是三种归一化中最简单的:
#   - 前向: 2 次遍历 (sum(x²) → normalize)，无权重乘法
#   - 反向: 无需并行归约 (没有可学习参数 w/b 要计算梯度)
#
# 参考: flash-linear-attention/fla/modules/l2norm.py


@triton.jit
def _l2_norm_fwd_fused(
    X,      # 输入指针
    Y,      # 输出指针
    Rstd,   # 1/||x||₂ 输出指针
    stride, # 行步长
    N,      # 列数
    eps,    # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    """
    L2Norm 前向 kernel — 每个程序实例处理一行

    对比三种归一化的 kernel 差异:
      LayerNorm: 遍历1(sum_x) → 遍历2(sum_sq) → 遍历3(y = w*(x-mean)*rstd + b)
      RMSNorm:   遍历1(sum_sq) → 遍历2(y = w * x * rstd)
      L2Norm:    遍历1(sum_sq) → 遍历2(y = x * rstd)  ← 最简单
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # ---- 第一遍: 计算 sum(x²) ----
    # 注意: 不除以 N (与 RMSNorm 的唯一区别在这里!)
    _sqsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _sqsum += x * x
    sqsum = tl.sum(_sqsum, axis=0)
    # rstd = 1/√(sum(x²) + ε)
    # 对比: RMSNorm 是 1/√(sum(x²)/N + ε)
    rstd = 1 / tl.sqrt(sqsum + eps)
    tl.store(Rstd + row, rstd)
    # ---- 第二遍: y = x * rstd ----
    # 对比: RMSNorm 是 y = w * x * rstd (多一个权重乘法)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = x * rstd
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _l2_norm_bwd_fused(
    DX,     # 输入梯度指针
    DY,     # 输出梯度指针
    Y,      # 前向输出指针 (保存 y 而非 x，节省重算)
    Rstd,   # 1/||x||₂ 指针
    stride, # 行步长
    N,      # 列数
    BLOCK_SIZE: tl.constexpr,
):
    """
    L2Norm 反向 kernel — dx = rstd * (dy - y * Σ(dy·y))

    对比三种归一化的反向:
      LayerNorm: dx = rstd * (w·dy - x̂·c₁ - c₂)  ← c₁ 和 c₂ 两个常数
      RMSNorm:   dx = rstd * (w·dy - x̂·c₁)        ← 只有 c₁
      L2Norm:    dx = rstd * (dy - y·dot)           ← 最简单，无权重，无需累积 dw/db

    注意: L2Norm 反向不需要并行归约策略 (因为没有可学习参数 w/b 要计算梯度)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    Y += row * stride
    DY += row * stride
    DX += row * stride
    # 加载数据
    y = tl.load(Y + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row)
    # dx = rstd * (dy - y * sum(dy * y))
    y = tl.where(mask, y, 0.)
    dy = tl.where(mask, dy, 0.)
    dot = tl.sum(dy * y, axis=0)
    dx = (dy - y * dot) * rstd
    tl.store(DX + cols, dx, mask=mask)


# ---- L2Norm autograd 包装 ----

class TritonL2Norm(torch.autograd.Function):
    """Triton L2Norm 的 autograd 包装 (比 LayerNorm/RMSNorm 简单: 无可学习参数)"""

    @staticmethod
    def forward(ctx, x, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("此 L2Norm 不支持特征维度 >= 64KB。")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _l2_norm_fwd_fused[(M,)](
            x_arg, y, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        # 保存 y (而非 x) 用于反向 — 因为 dx 公式中用到 y 和 rstd
        ctx.save_for_backward(y, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = torch.empty_like(dy)
        y_arg = y.reshape(-1, y.shape[-1])
        M, N = y_arg.shape
        _l2_norm_bwd_fused[(M,)](
            dx, dy, y, rstd,
            y_arg.stride(0), N,
            BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
        # 只返回 dx 和 eps 的 None (L2Norm 无可学习参数，反向最简单)
        return dx, None


triton_l2_norm = TritonL2Norm.apply


def test_l2_norm(M, N, dtype, eps=1e-6, device=DEVICE):
    """测试 L2Norm: Triton 实现 vs PyTorch 参考实现 (前向 + 反向)"""
    # 创建数据 (注意: 先创建 x 再设 requires_grad，使其为叶子张量)
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # Triton 前向 + 反向
    y_tri = triton_l2_norm(x, eps)
    y_tri.backward(dy, retain_graph=True)
    dx_tri = x.grad.clone()
    x.grad = None
    # PyTorch 参考前向 + 反向
    y_ref = pytorch_l2_norm(x, eps)
    y_ref.backward(dy, retain_graph=True)
    dx_ref = x.grad.clone()
    # 比较
    fwd_diff = (y_tri - y_ref).abs().max().item()
    dx_diff = (dx_tri - dx_ref).abs().max().item()
    print(f"  前向 max_diff = {fwd_diff:.2e}", end="")
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    print(" ✓")
    print(f"  dx   max_diff = {dx_diff:.2e}", end="")
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    print(" ✓")


# %%
# ========================
# 第五部分: 综合精度验证
# ========================

print("=" * 70)
print(" LayerNorm / RMSNorm / L2Norm 综合精度验证")
print("=" * 70)

M_test, N_test = 1151, 8192
dtype_test = torch.float16

print(f"\n>>> shape = ({M_test}, {N_test}), dtype = {dtype_test}")
print("-" * 50)

print(f"\n[1] LayerNorm — Triton vs torch.nn.functional")
test_layer_norm(M_test, N_test, dtype_test)
print("  ✓ 通过!")

print(f"\n[2] LayerNorm — PyTorch 手动实现 vs torch.nn.functional")
test_pytorch_layer_norm(M_test, N_test, dtype_test)

print(f"\n[3] RMSNorm — Triton vs PyTorch 参考")
test_rms_norm(M_test, N_test, dtype_test)

print(f"\n[4] L2Norm — Triton vs PyTorch 参考")
test_l2_norm(M_test, N_test, dtype_test)

print("\n" + "=" * 70)
print(" 全部精度验证通过!")
print("=" * 70)

# bench_layer_norm.run(save_path='.', print_data=True)


# %%
# ========================
# 总结
# ========================
#
# 本教程实现了三种常见归一化方法的 Triton kernel，并与纯 PyTorch 实现进行了精度对比。
#
# 一、数学公式对比
# ┌───────────┬──────────────────────────────────────────────┐
# │ 方法      │ 公式                                         │
# ├───────────┼──────────────────────────────────────────────┤
# │ LayerNorm │ y = w * (x - mean) / √(var + ε) + b         │
# │ RMSNorm   │ y = w * x / √(mean(x²) + ε)                 │
# │ L2Norm    │ y = x / √(sum(x²) + ε)                      │
# └───────────┴──────────────────────────────────────────────┘
#
# 二、区别
#   - LayerNorm: 完整归一化 = 去中心化 + 方差缩放 + 仿射变换 (w, b)
#   - RMSNorm:   简化版 = RMS缩放 + 权重 (仅 w)，去掉了均值和偏置
#   - L2Norm:    最简版 = L2范数缩放，无可学习参数
#
# 三、Triton 实现对比
# ┌───────────┬──────────┬────────────────────────┬────────────────────────┐
# │ 方法      │ 前向遍历 │ 反向计算               │ 梯度累积               │
# ├───────────┼──────────┼────────────────────────┼────────────────────────┤
# │ LayerNorm │ 3 次     │ dx 含 c₁ 和 c₂ 两项    │ 需要并行归约 dw, db    │
# │ RMSNorm   │ 2 次     │ dx 仅含 c₁ 项          │ 需要并行归约 dw        │
# │ L2Norm    │ 2 次     │ dx 用 dot(dy,y)        │ 无需归约 (无参数)      │
# └───────────┴──────────┴────────────────────────┴────────────────────────┘
#
# 四、性能特点
#   1. LayerNorm 前向需要 3 次遍历数据 (mean → var → normalize)，反向最复杂
#   2. RMSNorm 省去均值计算，前向 2 次遍历，反向比 LayerNorm 少一个 c₂ 项
#   3. L2Norm 最简单，无可学习参数，反向无需并行归约策略
#
# 五、实际应用
#   - LayerNorm: 传统 Transformer (BERT, GPT-2, T5, ViT)
#   - RMSNorm:   现代大语言模型 (LLaMA, Gemma, Qwen, DeepSeek, Mistral)
#   - L2Norm:    特征归一化、向量检索、对比学习, linear attention 中的 QK 归一化
#
# 六、关键实现细节
#   - 所有 kernel 均使用 float32 进行中间计算，避免 fp16/bf16 精度溢出
#   - LayerNorm/RMSNorm 的 dw/db 梯度使用锁 (atomic_cas) 实现并行归约
#   - L2Norm 反向保存 y (而非 x) 用于计算梯度，因为 dx = rstd*(dy - y*dot(dy,y))
#   - BLOCK_SIZE 限制为 64KB/element_size，确保数据能放入单个 kernel 的寄存器


# %%
# 参考文献
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
#
# .. [ZS2019] Biao Zhang and Rico Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
