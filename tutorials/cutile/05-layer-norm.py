# copy from https://github.com/NVIDIA/cutile-python/blob/main/test/kernels/layer_norm.py

import torch
import torch.nn.functional as F
import cuda.tile as ct
import math


import cuda.tile as ct

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def layer_norm_fwd(X, W, B, Y, Mean, Rstd, eps, TILE_N: ConstInt):
    """
    Layer Normalization 前向。

    分三阶段:
    1. 沿 N 维度累加求均值 mean
    2. 计算方差 var 和倒数标准差 rstd = 1/sqrt(var + eps)
    3. 归一化 + 仿射变换: Y = (X - mean) * rstd * W + B

    每个 block 负责一行 (bid_m), 行内按 TILE_N 分块迭代。
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    # 阶段 1: 计算均值
    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)

    # 阶段 2: 计算方差和倒数标准差
    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        # 仅对有效列应用 (X - mean), 超出 N 的 padding 部分置零避免污染方差
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx ** 2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)

    # 阶段 3: 归一化 + 仿射变换
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


def bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N):
    """反向传播辅助函数: 加载数据并计算 xhat 和 wdy。"""
    tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
    tdy = ct.load(DY, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    xhat = (tx - mean) * rstd
    wdy = tw * tdy
    # 对超出 N 的 padding 列置零
    mask = j * TILE_N + ct.arange(TILE_N, dtype=ct.int32) < N
    xhat = ct.where(mask, xhat, 0)
    wdy = ct.where(mask, wdy, 0)
    return tdy, xhat, wdy


@ct.kernel
def layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, TILE_N: ConstInt):
    """
    反向传播 Part 1: 计算 dX 并累加部分 dW/dB。

    每个 block 处理一行的 dX, 同时将本行对 dW/dB 的贡献累加到
    GROUP_SIZE_M 个分组中。分组间通过原子锁 (CAS) 实现互斥访问。

    dX = rstd * (wdy - xhat * mean(xhat * wdy) - mean(wdy))
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]
    GROUP_SIZE_M = DW.shape[0]
    group_bid_m = bid_m % GROUP_SIZE_M

    mean = ct.load(Mean, index=(bid_m,), shape=(1,))
    rstd = ct.load(Rstd, index=(bid_m,), shape=(1,))

    # 先遍历所有 tile 累加 c1 = mean(xhat * wdy), c2 = mean(wdy)
    c1 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    c2 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 += xhat * wdy
        c2 += wdy
    c1 = ct.sum(c1, axis=1) / N
    c2 = ct.sum(c2, axis=1) / N

    # 计算 dX 并累加部分 dW/dB
    for j in range(num_tiles):
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy - (xhat * c1 + c2)) * rstd
        ct.store(DX, index=(bid_m, j), tile=tdx.astype(DX.dtype))

        partial_dw = (tdy * xhat).astype(DW.dtype)
        partial_db = tdy.astype(DB.dtype)

        # 原子锁: acquire 语义确保读到最新的 partial sum
        while ct.atomic_cas(Locks, group_bid_m, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
            pass

        # 读-改-写: 将本行贡献累加到分组的 partial sum
        partial_dw += ct.load(DW, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        partial_db += ct.load(DB, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        ct.store(DW, index=(group_bid_m, j), tile=partial_dw)
        ct.store(DB, index=(group_bid_m, j), tile=partial_db)

        # release 语义确保写入对后续 acquire 可见
        ct.atomic_xchg(Locks, group_bid_m, 0, memory_order=ct.MemoryOrder.RELEASE)


@ct.kernel
def layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    反向传播 Part 2: 将 GROUP_SIZE_M 个分组的部分梯度沿 M 维度归约, 得到最终的 dW 和 dB。
    """
    bid_n = ct.bid(0)
    num_tiles = ct.num_tiles(DW, axis=0, shape=(TILE_M, TILE_N))

    dw = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    db = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for i in range(num_tiles):
        dw += ct.load(DW, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
        db += ct.load(DB, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
    sum_dw = ct.sum(dw, axis=0)
    sum_db = ct.sum(db, axis=0)

    ct.store(FINAL_DW, index=(bid_n,), tile=sum_dw.astype(FINAL_DW.dtype))
    ct.store(FINAL_DB, index=(bid_n,), tile=sum_db.astype(FINAL_DB.dtype))


# --- CuTile LayerNorm 包装 ---

class CuTileLayerNorm(torch.autograd.Function):
    """封装 cuTile LayerNorm 前向和反向为 PyTorch autograd Function。"""

    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        """前向: 展平输入, 启动 layer_norm_fwd kernel, 保存中间结果供反向使用。"""
        x = input.reshape(-1, input.shape[-1])
        y = torch.empty_like(x)
        M, _ = x.shape

        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        TILE_N = 1024
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_fwd,
                  (x, weight, bias, y, mean, rstd, eps, TILE_N))

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.TILE_N = TILE_N

        return y.reshape(*input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向分两步:
        1. layer_norm_bwd_dx_partial_dwdb: 计算 dX 并将 dW/dB 按组累加
        2. layer_norm_bwd_dwdb: 对分组结果做最终归约得到 dW 和 dB
        """
        x, weight, bias, mean, rstd = ctx.saved_tensors
        TILE_N = ctx.TILE_N
        M, N = x.shape
        GROUP_SIZE_M = 64

        dy = grad_output.reshape(-1, grad_output.shape[-1])
        dx = torch.empty_like(dy)

        # 分组累加缓冲区 (GROUP_SIZE_M 个分组, 每组长度为 N)
        dw = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=weight.device)
        db = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=bias.device)
        locks = torch.zeros(GROUP_SIZE_M, dtype=torch.int32, device=weight.device)

        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_bwd_dx_partial_dwdb,
                  (dx, dy, dw, db, x, weight, mean, rstd, locks, TILE_N))

        # 最终归约
        final_dw = torch.empty((N,), dtype=weight.dtype, device=weight.device)
        final_db = torch.empty((N,), dtype=bias.dtype, device=bias.device)
        TILE_M = 32

        ct.launch(torch.cuda.current_stream(), (math.ceil(N / TILE_N),), layer_norm_bwd_dwdb,
                  (dw, db, final_dw, final_db, TILE_M, TILE_N))

        return dx.reshape(*grad_output.shape), final_dw, final_db, None


def cutile_layer_norm(x, weight, bias, eps):
    return CuTileLayerNorm.apply(x, weight, bias, eps)


DEVICE = torch.cuda.current_device()

def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    """正确性测试: 比较 cuTile LayerNorm 与 PyTorch 原生实现的前向和反向结果。"""
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    y_tri = cutile_layer_norm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
    print("✅ cuTile and Torch match")

test_layer_norm(1151, 8192, torch.float16)


import triton
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import importlib.util
_spec = importlib.util.spec_from_file_location("triton_05_layer_norm", os.path.join(os.path.dirname(__file__), "../triton/05-layer-norm.py"))
_triton_layernorm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_triton_layernorm_module)
HAS_APEX = _triton_layernorm_module.HAS_APEX
triton_laynrom = _triton_layernorm_module.layer_norm

configs = []

for mode in ['fwd', 'bwd']:
    configs.append(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 32)],
            line_arg='provider',
            line_vals=['cutile', 'triton', 'torch'] + (['apex'] if HAS_APEX else []),
            line_names=['cuTile', 'Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel='GB/s',
            plot_name=f'layer-norm-{mode}',
            args={'M': 4096, 'dtype': torch.float16, 'mode': mode},
        ))

@triton.testing.perf_report(configs)
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
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
            return triton_laynrom(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

        if provider == "cutile":
            return cutile_layer_norm(x, weight, bias, eps)  # noqa: F811, E704

    if mode == 'fwd':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    if mode == 'bwd':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


bench_layer_norm.run(show_plots=True, print_data=True)
