## Triton-03-Matrix Multiplication

### 前言

本系列教程记录了学习 Triton 的笔记，参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)，翻译并添加了一些自己的理解。

- Triton-01-Vector Addition
- Triton-02-Fused Softmax
- **Triton-03-Matrix Multiplication**（本文）
- Triton-04-Low-Memory Dropout
- Triton-05-Layer Normalization
- Triton-06-Fused Attention
- Triton-07-Extern Functions
- Triton-08-Grouped GEMM
- Triton-09-Persistent Matmul
- Triton-10-Block-Scaled Matmul
- Triton-11-Programmatic Dependent Launch

矩阵乘法
=====================
在本教程中，你将编写一个非常简短的高性能 FP16 矩阵乘法kernel，其性能可以与 cuBLAS 或 rocBLAS 相媲美。

你将具体学习到:

* 块级矩阵乘法。

* 多维指针算术运算。

* 通过程序重排序来提高 L2 缓存命中率。

* 自动性能调优。

个块是并行计算的，为什么顺序会影响？三个原因:
- 有限的SM数量: 虽然逻辑上"并行",但物理上只有有限的SM, 块会被动态调度
- 调度器的行为: GPU调度器通常按pid顺序分配块,因此改变pid映射可以影响哪些块"在时间上相邻".
- 共享的L2缓存: 所有SM共享L2缓存, 但时间上相邻执行的块会竞争缓存空间.


GROUP_SIZE_M 的作用:
不是让块"真正并行"(GPU硬件决定并行度),
而是让"时间上相邻执行"的块也"空间上访问相邻数据"
从而提高L2缓存命中率.

### 动机

矩阵乘法是大多数现代高性能计算系统的关键构建块。
它们的优化非常困难，因此其实现通常由硬件供应商自己完成，作为所谓的"kernel库"
（例如 cuBLAS）的一部分。
不幸的是，这些库通常是专有的，无法轻松定制以适应现代深度学习工作负载的需求
（例如，融合激活函数）。
在本教程中，你将学习如何使用 Triton 自己实现高效的矩阵乘法，
并且这种方式易于定制和扩展。

粗略地说，我们将要编写的kernel将实现以下分块算法来执行 (M, K) 矩阵与 (K, N) 矩阵的乘法:

```python
# 并行执行
for m in range(0, M, BLOCK_SIZE_M):
  # 并行执行
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```

其中，双重嵌套 for 循环的每次迭代都由一个专用的 Triton 程序实例执行。

### 计算kernel

上述算法实际上在 Triton 中实现起来相当简单。
主要的困难来自于在内部循环中计算必须读取 :code:`A` 和 :code:`B` 块的内存位置。
为此，我们需要多维指针算术运算。

#### 指针算术运算

对于行优先的 2D tensor :code:`X`，:code:`X[i, j]` 的内存位置由
:code:`&X[i, j] = X + i*stride_xi + j*stride_xj` 给出。
因此，:code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` 和
:code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` 的指针块可以在伪代码中定义为:

```python
&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
&B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
```

这意味着 A 和 B 块的指针可以在 Triton 中初始化（即 :code:`k=0`），如以下代码所示。
还要注意，我们需要额外的取模运算来处理 :code:`M` 不是 :code:`BLOCK_SIZE_M` 的倍数
或 :code:`N` 不是 :code:`BLOCK_SIZE_N` 的倍数的情况，在这种情况下，我们可以用一些无用的值填充数据，
这些值不会对结果产生影响。对于 :code:`K` 维度，我们稍后将使用掩码加载语义来处理。

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
```

然后在内部循环中更新如下:

```python
a_ptrs += BLOCK_SIZE_K * stride_ak;
b_ptrs += BLOCK_SIZE_K * stride_bk;
```

#### L2 缓存优化

如上所述，每个程序实例计算 :code:`C` 的一个 :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]` 块。
重要的是要记住，这些块的计算顺序很重要，因为它会影响我们程序的 L2 缓存命中率，
不幸的是，简单的行优先顺序:

```python
pid = tl.program_id(axis=0)
grid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // grid_n
pid_n = pid % grid_n
```

是无法满足要求的。

一个可能的解决方案是按照促进数据重用的顺序启动块。
这可以通过在切换到下一列之前将块"超级分组"为 :code:`GROUP_M` 行来实现:

```python
# 程序 ID
pid = tl.program_id(axis=0)
# 沿 M 轴的程序 ID 数量
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# 沿 N 轴的程序 ID 数量
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
# 组中的程序数量
num_pid_in_group = GROUP_SIZE_M * num_pid_n
# 此程序所在组的 ID
group_id = pid // num_pid_in_group
# 组中第一个程序的行 ID
first_pid_m = group_id * GROUP_SIZE_M
# 如果 `num_pid_m` 不能被 `GROUP_SIZE_M` 整除，则最后一组较小
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# *在组内*，程序按列优先顺序排列
# *启动网格*中程序的行 ID
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
# *启动网格*中程序的列 ID
pid_n = (pid % num_pid_in_group) // group_size_m
```

例如，在以下矩阵乘法中，每个矩阵是 9 个块乘 9 个块，
我们可以看到，如果我们按行优先顺序计算输出，我们需要将 90 个块加载到 SRAM（L2缓存） 中
才能计算前 9 个输出块，但如果我们按分组顺序执行，我们只需要加载 54 个块。
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  .. image:: grouped_vs_row_major_ordering.png
注意: 官方图不是展示完整的一组，而是展示计算前9个输出块时的对比！

    9×9 的块网格，GROUP_SIZE_M = 3
    其中每个格子是一个程序,每个程序实例内部处理 BLOCK_SIZE_M×BLOCK_SIZE_N 个元素

                  9个块列 (num_pid_n)
         ┌───────────────────────────────────┐
         ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐  ┐
块行0     │   │   │   │   │   │   │   │   │   │  │
         ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  ｜ 组0
块行1     │   │   │   │   │   │   │   │   │   │  │ (3个块行)
         ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  ｜
块行2     │   │   │   │   │   │   │   │   │   │  ┘
         ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  ┐
块行3     │   │   │   │   │   │   │   │   │   │  │
         ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  ｜ 组1
块行4     │   │   │   │   │   │   │   │   │   │  │ (3个块行)
         ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  ｜
块行5     │   │   │   │   │   │   │   │   │   │  ┘
         └───┴───┴───┴───┴───┴───┴───┴───┴───┘

# 示例 # #
pid = 5  # 当前程序ID
# 第1步: 一个组有多少个块？
num_pid_in_group = GROUP_SIZE_M * num_pid_n = 3 * 9 = 27
# 第2步: 当前程序属于哪个组？
group_id = pid // num_pid_in_group = 5 // 27 = 0  (属于第0组)
# 组中第一个程序的行 ID
first_pid_m = group_id * GROUP_SIZE_M = 0 * 3 = 0
# 如果 `num_pid_m` 不能被 `GROUP_SIZE_M` 整除，则最后一组较小, 防止越界
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) = min(9 - 0, 3) = 3
# 第3步: 转换为二维坐标 (列优先!)
# 行索引 (position对行数取模)
pid_m = first_pid_m + (pid % group_size_m) = 0 + (5 % 3) = 2
# 列索引 (position除以行数)
pid_n = (pid % num_pid_in_group) // group_size_m = (5 % 27) // 3 = 2
结果: pid=5 负责计算第 C[2, 1] 块 ✓

   列0 列1 列2 列3 列4 列5 列6 列7 列8
   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐
行0 │0 │3 │6 │9 │12│15│18│21│24│
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┤
行1 │1 │4 │7 │10│13│16│19│22│25│
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┤
行2 │2 │5 │8 │11│14│17│20│23│26│
   └──┴──┴──┴──┴──┴──┴──┴──┴──┘
   └────────┘
   官方图只展示了这一小部分

在实践中，这可以在某些硬件架构上将我们的矩阵乘法kernel的性能提高超过 10%
（例如，在 A100 上从 220 提高到 245 TFLOPS）。

### 最终结果

```python
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # 适合 fp8 输入的良好配置
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit` 装饰的函数可以通过使用 `triton.autotune` 装饰器进行自动调优，它需要:
#   - 一个 `triton.Config` 对象列表，定义要尝试的元参数（例如 `BLOCK_SIZE_M`）和编译选项（例如 `num_warps`）的不同配置
#   - 一个自动调优*键*，其值的变化将触发对所有提供的配置的评估
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # 矩阵指针
        a_ptr, b_ptr, c_ptr,
        # 矩阵维度
        M, N, K,
        # stride 变量表示在特定维度上移动 1 个元素时指针增加多少。
        # 例如，`stride_am` 是向下移动一行（A 有 M 行）时 `a_ptr` 增加的量。
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # 元参数
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """用于计算矩阵乘法 C = A x B 的kernel。
    A 的形状为 (M, K), B 的形状为 (K, N), C 的形状为 (M, N)
    """
    # -----------------------------------------------------------
    # 将程序 ID `pid` 映射到它应该计算的 C 块。
    # 这以分组顺序完成，以促进 L2 数据重用。
    pid = tl.program_id(axis=0)
    # 沿 M 轴的程序 ID 数量
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # 沿 N 轴的程序 ID 数量
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # 组中的程序数量
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # 此程序所在组的 ID
    group_id = pid // num_pid_in_group
    # 组中第一个程序的行 ID
    first_pid_m = group_id * GROUP_SIZE_M
    # 如果 `num_pid_m` 不能被 `GROUP_SIZE_M` 整除，则最后一组较小
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # *在组内*，程序按列优先顺序排列: 先往下走，再往右走
    # *启动网格*中程序的行 ID
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # *启动网格*中程序的列 ID
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # 添加一些整数边界假设。
    # 这有助于指导后端中的整数分析以优化加载/存储偏移地址计算
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # 为 A 和 B 的第一个块创建指针。
    # 取模是为了处理边界问题
    # 我们将在 K 方向上移动并累积时推进此指针
    # 有关详细信息，请参见上面的 `指针算术运算` 部分
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # `a_ptrs` 是一个 [BLOCK_SIZE_M, BLOCK_SIZE_K] 指针块
    # `b_ptrs` 是一个 [BLOCK_SIZE_K, BLOCK_SIZE_N] 指针块
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # 迭代计算 C 矩阵的一个块，每个块需要累积K维度。
    # 我们累积到一个 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 的 fp32 值块中以获得更高的精度。
    # `accumulator` 将在循环后转换回 fp16。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 和 B 的下一个块，通过检查 K 维度生成掩码。
        # 如果超出边界，则将其设置为 0。
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # 我们沿 K 维度累积。
        accumulator = tl.dot(a, b, accumulator)
        # 将指针推进到下一个 K 块。
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # 你可以在这里融合任意激活函数
    # 当累加器仍然是 FP32 时！
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # 使用掩码写回输出矩阵 C 的块。
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# 我们可以通过在 `matmul_kernel` 中将 `leaky_relu` 作为 `ACTIVATION` 元参数提供来融合它。
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# 现在我们可以创建一个方便的包装函数，它只接受两个输入tensor，
# 并且 (1) 检查任何形状约束；(2) 分配输出；(3) 启动上述kernel。

def matmul(a, b, activation=""):
    # 检查约束。
    assert a.shape[1] == b.shape[0], "不兼容的维度"
    assert a.is_contiguous(), "矩阵 A 必须是连续的"
    M, K = a.shape
    K, N = b.shape
    # 分配输出。
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D 启动kernel，其中每个块都有自己的程序。
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


### 单元测试

我们可以针对原生 torch 实现（即 cuBLAS）测试我们的自定义矩阵乘法操作。

```python
torch.manual_seed(0)
a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton 和 Torch 匹配")
else:
    print("❌ Triton 和 Torch 不同")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # 为了效率预先转置 b。
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton 和 Torch 匹配")
    else:
        print("❌ Triton 和 Torch 不同")
```

### 基准测试

方阵性能
~~~~~~~~~~~~~~~~~~~~~~~~~~

现在我们可以将我们的kernel性能与 cuBLAS 或 rocBLAS 的性能进行比较。
这里我们关注方阵，但可以随意调整此脚本以对任何其他矩阵形状进行基准测试。

```python
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # 用作绘图 x 轴的参数名称
            x_vals=[128 * i for i in range(2, 33)],  # `x_name` 的不同可能值
            line_arg="provider",  # 其值对应于绘图中不同线条的参数名称
            # `line_arg` 的可能值
            # 对于 fp8 情况，不要与 cublas 比较，因为 torch.matmul 目前不支持 fp8。
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # 线条的标签名称
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # 线条样式
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # y 轴的标签名称
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # 绘图的名称，也用作保存绘图的文件名。
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
```
