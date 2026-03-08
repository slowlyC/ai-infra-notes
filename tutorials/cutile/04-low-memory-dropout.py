import torch
import cuda.tile as ct
import math


@ct.kernel
def dropout_kernel(
    x,
    x_keep,
    output,
    p: ct.Constant[float],    # dropout 概率 (丢弃比例)
    TILE: ct.Constant[int],   # 每个 block 处理的元素数
):
    """
    低显存 Dropout kernel。

    采用 inverted dropout: 对保留的元素除以 (1-p),
    使得训练和推理时期望值一致, 推理时无需额外缩放。
    x_keep 是预先生成的 bool mask (True 表示保留)。
    """
    bid = ct.bid(0)

    # 加载当前 block 负责的 tile
    x_tile = ct.load(x, index=(bid), shape=(TILE,))
    x_keep_tile = ct.load(x_keep, index=(bid), shape=(TILE,))

    # inverted dropout: 保留的元素除以 (1-p) 以保持期望值不变
    output_tile = ct.where(x_keep_tile, x_tile / (1 - p), 0.0)

    ct.store(output, index=(bid,), tile=output_tile)


def dropout(x, x_keep, p):
    """Dropout 主机端包装函数, 计算 grid 尺寸并启动 kernel。"""
    output = torch.empty_like(x)
    assert x.is_contiguous()
    N = x.numel()
    TILE = 1024
    grid = (math.ceil(N / TILE), 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, dropout_kernel, (x, x_keep, output, p, TILE))
    return output


# %%
# 正确性验证: 与 torch 参考实现对比
# -----------------------------------

DEVICE = torch.cuda.current_device()
N = 98432
x = torch.randn(size=(N, ), device=DEVICE)
p = 0.5
x_keep = torch.rand(size=(N, ), device=DEVICE) > p
cutile_output = dropout(x, x_keep=x_keep, p=p)

# torch 参考实现 (等价于 inverted dropout)
torch_output = torch.where(x_keep, x / (1 - p), torch.zeros_like(x))

if torch.allclose(cutile_output, torch_output, atol=1e-2, rtol=0):
    print("✅ cuTile and Torch match")
else:
    print("❌ cuTile and Torch differ")

# 小规模示例: 打印 input / keep mask / output 对照表
N = 10
x = torch.randn(size=(N, ), device=DEVICE)
x_keep = (torch.rand(size=(N, ), device=DEVICE) > p).to(torch.bool)
output = dropout(x, x_keep=x_keep, p=p)
import tabulate

print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

# 注: Triton 中有 tl.rand 和 seeded_dropout 可在 kernel 内直接生成随机数
# (基于 Philox PRNG), 而 cuTile 目前需要外部预生成 mask。
# 参考: https://github.com/triton-lang/triton/blob/main/python/triton/language/random.py
