# Triton cumsum Kernel 访存优化

本文记录了对 FLA（Flash Linear Attention）中 `chunk_local_cumsum_scalar_kernel` 的性能分析和优化过程。背景是对 Qwen3.5/Qwen3-NEXT 模型中 GDN kernel 进行 NCU profiling 发现该 kernel 存在访存未合并问题后，通过将 1D scan 改为 2D scan 引导编译器生成向量化访存指令，长序列下获得约 5 倍加速。

关于 `tl.cumsum` 从 Python 到 PTX 的完整编译流程分析，参见 [Triton Scan Op 编译全流程](https://zhuanlan.zhihu.com/p/2007221164263624909)。相关代码已开源至Sglang。

## 一、背景：FLA 中的 cumsum 操作

在 Flash Linear Attention 中，`chunk_local_cumsum` 用于对 gate tensor `g` 做 chunk 级别的累积求和。根据输入维度不同，分为 scalar（3D: `[B, T, H]`）和 vector（4D: `[B, T, H, S]`）两种情况。

入口函数根据 tensor 维度自动分发：

```python
@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype)
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype)
```

本文聚焦 **scalar cumsum** 的优化（3D 输入，`[B, T, H]` 或 `[B, H, T]`）。

## 二、问题发现：NCU 性能分析

通过 NCU profiling 发现以下问题：

- 存在大量未合并访存
- 未使用 shared memory，问题与 L1/L2 cache 相关
- `Sectors/Req = 1`，说明存在访存不对齐或未合并问题

对于全局内存访问，一个 warp 的 32 个线程同时发出的访存请求会被合并为尽量少的内存事务。理想情况下，32 个线程访问连续的 128 字节（32 x 4 bytes），只需 1 次 128-byte 事务（4 个 sector）。当 `Sectors/Req = 1` 时，意味着每个请求只传输了 1 个 32-byte sector，说明线程间的地址分散在不同的 cache line 上。

## 三、原始 Kernel 分析

### 3.1 原始 Kernel 代码

```python
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
```

Host 侧调用：

```python
grid = (NT, B * H)
chunk_local_cumsum_scalar_kernel[grid](
    s=g_org, o=g, scale=scale,
    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    T=T, B=B, H=H, BT=BT,
    HEAD_FIRST=head_first, REVERSE=reverse,
    HAS_SCALE=scale is not None, IS_VARLEN=cu_seqlens is not None,
    num_warps=8, num_stages=3,
)
```

### 3.2 TTIR 分析

原始 kernel 生成的 TTIR 中，`tt.scan` 操作在 1D tensor 上执行：

```mlir
%25 = "tt.scan"(%24) <{axis = 0 : i32, reverse = false}> ({
^bb0(%arg3: f32, %arg4: f32):
  %28 = arith.addf %arg3, %arg4 : f32
  tt.scan.return %28 : f32
}) : (tensor<64xf32>) -> tensor<64xf32>
```

`tl.cumsum` 被降级为 `tt.scan` 操作，combine region 中的 `arith.addf` 即为加法组合函数。输入输出都是 `tensor<64xf32>`（1D tensor，对应 `BT=64`）。

### 3.3 TTGIR 与 Layout 分析

TTGIR 中自动添加的 blocked layout：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [8],
  order = [0]
}>

module attributes {
  "ttg.num-ctas" = 1, "ttg.num-warps" = 8,
  ttg.target = "cuda:80", "ttg.threads-per-warp" = 32
}
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `sizePerThread` | [1] | 每个线程处理 1 个元素 |
| `threadsPerWarp` | [32] | 一个 warp 的 32 个线程各处理 1 个元素 |
| `warpsPerCTA` | [8] | 8 个 warp，共 256 个线程处理 64 个元素 |

`tensor<64xf32>` 由 256 个线程处理，实际上只有前 64 个线程有数据，其余 192 个线程空闲。同时 8 个 warp 意味着 scan 需要跨 warp 协调（shared memory + barrier）。

### 3.4 访存未合并的根因

先看两种 layout 下 `make_block_ptr` 的含义。

**HEAD_FIRST=False**（`g.shape = [B, T, H]`，按列取 BT 块）：

```
# g.shape = [T, H], 取第 i_h 列的 BT 个元素
#     h0  h1  h2  h3  h4  ← H 维度
# t0 [ .   .   X   .   . ]
# t1 [ .   .   X   .   . ]
# t2 [ .   .   .   .   . ]  ← 取 i_h=2 列
# t3 [ .   .   .   .   . ]
# ↑
# T 维度

p_s = tl.make_block_ptr(
    s + bos * H + i_h,   # base: 跳到第 i_h 列
    (T,),                 # shape: T 个元素
    (H,),                 # strides: 步长为 H（跨行取数据）
    (i_t * BT,),          # offsets: 从第 i_t*BT 行开始
    (BT,),                # block_shape
    (0,),                 # order
)
```

等价的 offset 写法：

```python
row_indices = i_t * BT + tl.arange(0, BT)
offsets = row_indices * H + i_h    # 每行间隔 H 个元素
mask = row_indices < T
b_s = tl.load(s + bos * H + offsets, mask=mask, other=0.0)
```

**HEAD_FIRST=True**（`g.shape = [B, H, T]`，按行取 BT 块）：

```
# g.shape = [H, T], 取第 i_h 行的 BT 个元素
#     t0  t1  t2  t3  t4  ← T 维度
# h0 [ .   .   .   .   . ]
# h1 [ .   .   .   .   . ]
# h2 [ X   X   .   .   . ]  ← 取 i_h=2 行
# h3 [ .   .   .   .   . ]
# ↑
# H 维度

p_s = tl.make_block_ptr(
    s + bos * H + i_h * T,  # base: 跳到第 i_h 行起始
    (T,),                    # shape
    (1,),                    # strides: 步长为 1（行内连续）
    (i_t * BT,),             # offsets
    (BT,),                   # block_shape
    (0,),                    # order
)
```

`HEAD_FIRST=True` 时 stride=1，数据连续，不存在访存合并问题。问题出在 `HEAD_FIRST=False` 时 stride=H。

具体来说，输入 Tensor 的内存布局是 `[B, T, H]`，cumsum 在 T 维度累加。同一个 head 的相邻时间步在内存中间隔 H 个元素：

```
内存布局: [t0_h0, t0_h1, ..., t0_h31, t1_h0, t1_h1, ..., t1_h31, ...]
                                       ↑ stride = H = 32

Thread 0:  读取 t0_h0 (地址 base + 0 * 32 * 4 = base + 0)
Thread 1:  读取 t1_h0 (地址 base + 1 * 32 * 4 = base + 128)
Thread 2:  读取 t2_h0 (地址 base + 2 * 32 * 4 = base + 256)
...
Thread 31: 读取 t31_h0 (地址 base + 31 * 32 * 4 = base + 3968)
```

H = 32 时，相邻线程的地址间距 = 128 bytes = 1 个完整 cache line。每个线程读不同 cache line：
- 32 个线程 → 32 次内存事务
- 每次事务传输 32 字节（1 sector），但只用 4 字节（1 个 float）
- 带宽利用率 = 4/32 = 12.5%

### 3.5 PTX 验证

从编译生成的 PTX 代码可以确认问题：

```ptx
.reqntid 256, 1, 1     // 256 个线程 = 8 warps

// 标量加载 (每次只读 1 个 float)
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];

// warp 内 scan (Kogge-Stone 算法)
shfl.sync.up.b32    %r22, %r1, 1, 0, -1;    // offset=1，因为 1D layout threadStride=1
add.f32     %f3, %f1, %f2;
selp.f32     %f4, %f1, %f3, %p7;
// ... 重复 5 轮 (log2(32) = 5 步)

// 跨 warp 同步 (通过 shared memory)
@%p2 st.shared.b32 [ %r2 + 0 ], %r3;        // 写 shared memory
bar.sync     0;                                // barrier 同步
ld.shared.f32     %f17, [global_smem];        // 读 shared memory

// 标量存储
@%p3 st.global.b32 [ %rd2 + 0 ], { %r4 };
```

从 PTX 可以确认三个问题：（1）`ld/st.global.b32` 标量访存，没有向量化；（2）8 个 warp 需要 `st.shared` + `bar.sync` + `ld.shared` 做跨 warp 协调；（3）1D layout 的 `threadStride = 1`，warp scan 跑 5 轮 `shfl.sync.up`。

## 四、优化方案：向量化访存

### 4.1 优化思路

把 1D scan 改成 2D scan——增加一个 BH 维度（`BH = min(8, next_power_of_2(H))`），让每个 block 同时处理 BH 个 head。`tensor<BT xf32>` 变成 `tensor<BT x BH xf32>` 后，axis=1 方向的多个 head 在内存中连续排列，Triton 编译器会自动选择合适的 blocked layout 做向量化。通过 `FLA_CUMSUM_SCALAR_VECTORIZATION` 环境变量控制是否启用。

### 4.2 优化后 Kernel 代码

```python
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_vectorization_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    n_groups = tl.cdiv(H, BH)
    i_b, i_hg = i_bh // n_groups, i_bh % n_groups
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H, (H, T), (T, 1), (i_hg * BH, i_t * BT), (BH, BT), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o + bos * H, (H, T), (T, 1), (i_hg * BH, i_t * BT), (BH, BT), (1, 0)
        )
    else:
        p_s = tl.make_block_ptr(
            s + bos * H, (T, H), (H, 1), (i_t * BT, i_hg * BH), (BT, BH), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o + bos * H, (T, H), (H, 1), (i_t * BT, i_hg * BH), (BT, BH), (1, 0)
        )
    # [BT, BH]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if HEAD_FIRST:
        b_o = tl.cumsum(b_s, axis=1)
        if REVERSE:
            b_z = tl.sum(b_s, axis=1)
            b_o = -b_o + b_z[:, None] + b_s
    else:
        b_o = tl.cumsum(b_s, axis=0)
        if REVERSE:
            b_z = tl.sum(b_s, axis=0)
            b_o = -b_o + b_z[None, :] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
```

Host 侧调用：

```python
BH = min(8, triton.next_power_of_2(H))
grid = (NT, B * triton.cdiv(H, BH))
chunk_local_cumsum_scalar_vectorization_kernel[grid](
    s=g_org, o=g, scale=scale,
    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    T=T, B=B, H=H, BT=BT, BH=BH,
    HEAD_FIRST=head_first, REVERSE=reverse,
    HAS_SCALE=scale is not None, IS_VARLEN=cu_seqlens is not None,
    num_warps=4, num_stages=3,
)
```

### 4.3 优化对比

两个 kernel 的差异可以从以下几个层面理解：

#### Grid 与并行策略

| 维度 | 原始 kernel | 优化后 kernel |
|------|------------|--------------|
| grid.x | `NT`（chunk 数量） | `NT`（不变） |
| grid.y | `B * H`（每个 head 一个 block） | `B * cdiv(H, BH)`（多个 head 共享一个 block） |
| 每 block 处理 | 1 个 head 的 BT 个时间步 | BH 个 head 的 BT 个时间步 |

原始 kernel 每个 block 处理 1 个 head，所以 block 数量是 `B * H`。优化后每个 block 处理 BH 个 head（BH 通常为 8），block 数量减少为 `B * ceil(H/BH)`，但每个 block 内的数据更密集。

#### 取址操作(HEAD_FIRST=False）

原始 kernel：

```python
p_s = tl.make_block_ptr(
    s + bos * H + i_h,    # 基地址: 定位到特定 head
    (T,),                  # 逻辑形状: 1D, T 个元素
    (H,),                  # stride: 每步跳 H 个元素 (关键问题!)
    (i_t * BT,),           # 偏移
    (BT,),                 # block 大小
    (0,),                  # order
)
```

优化后 kernel：

```python
p_s = tl.make_block_ptr(
    s + bos * H,           # 基地址: 不定位到特定 head
    (T, H),                # 逻辑形状: 2D, T x H
    (H, 1),                # stride: 行步长=H, 列步长=1 (相邻 head 连续!)
    (i_t * BT, i_hg * BH), # 偏移: (时间步偏移, head group 偏移)
    (BT, BH),              # block 大小: BT x BH
    (1, 0),                # order: 列优先 (axis=1 为内层)
)
```

对比一下：原始 kernel 的 stride 是 `(H,)`，相邻线程地址间距 `H * sizeof(float)` bytes——每个线程读不同 cache line。优化后列 stride 变成 `(1,)`，BH 个 head 的数据在内存中连续排列，编译器能合并为向量化加载。

#### Scan 轴对比

| 场景 | 原始 kernel | 优化后 kernel |
|------|------------|--------------|
| HEAD_FIRST=True | `cumsum(b_s, axis=0)` on `[BT]` | `cumsum(b_s, axis=1)` on `[BH, BT]` |
| HEAD_FIRST=False | `cumsum(b_s, axis=0)` on `[BT]` | `cumsum(b_s, axis=0)` on `[BT, BH]` |

无论哪种场景，scan 都沿 T 维度执行。区别在于优化后 kernel 在 H 维度增加了并行度，使得每个线程可以一次加载多个 head 的数据。

## 五、编译结果深度分析

### 5.1 优化后 TTIR

```mlir
module {
  tt.func public @chunk_local_cumsum_scalar_vectorization_kernel(
      %s: !tt.ptr<f32>, %o: !tt.ptr<f32>, %T: i32) {

    // 地址计算: 2D block pointer
    %b_s = tt.splat %p_s_3 : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    // ... 行偏移 (stride=32) 和列偏移计算 ...
    %b_s_22 = arith.addi %b_s_15, %b_s_21 : tensor<64x8xi64>
    %b_s_23 = tt.addptr %b_s, %b_s_22 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi64>

    // boundary check: 行范围 [0, T) 和列范围 [0, 32)
    %b_s_33 = arith.andi %b_s_28, %b_s_32 : tensor<64x8xi1>

    // 加载 2D 数据
    %b_s_34 = tt.load %b_s_23, %b_s_33 : tensor<64x8x!tt.ptr<f32>>

    // scan 操作 (在 axis=0 即 T 维度上)
    %b_o = "tt.scan"(%b_s_34) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%b_o_35: f32, %b_o_36: f32):
      %b_o_37 = arith.addf %b_o_35, %b_o_36 : f32
      tt.scan.return %b_o_37 : f32
    }) : (tensor<64x8xf32>) -> tensor<64x8xf32>

    // 存储结果
    tt.store %6, %b_o, %b_s_33 : tensor<64x8x!tt.ptr<f32>>
  }
}
```

1D 和 2D TTIR 的差别只在 tensor 形状。但就是这个形状变化，在 TTGIR 阶段会推导出完全不同的 blocked layout。

### 5.2 优化后 TTGIR Layout

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1, 4],
  threadsPerWarp = [16, 2],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>

module attributes {
  "ttg.num-ctas" = 1, "ttg.num-warps" = 1,
  ttg.target = "cuda:90", "ttg.threads-per-warp" = 32
}
```

| 参数 | 原始 | 优化后 | 含义 |
|------|------|--------|------|
| `sizePerThread` | [1] | [1, 4] | 每线程从 1 个元素变为 4 个连续元素 |
| `threadsPerWarp` | [32] | [16, 2] | Warp 内线程从 1D 变为 2D 布局 |
| `warpsPerCTA` | [8] | [1, 1] | 从 8 个 warp 减少为 1 个 warp |
| `order` | [0] | [1, 0] | 新增列优先遍历 |

Layout 解读：
- 32 个线程按 16 行 x 2 列排列
- 每个线程处理 `1 x 4 = 4` 个连续元素（沿 axis=1 方向）
- 总处理量：`(16 x 1) x (2 x 4) = 16 x 8 = 128` 个元素
- 但 tensor 是 `64 x 8`，所以需要 4 个 "chunk"（`64 / 16 = 4`），每个 chunk 16 行

数据到线程的映射关系：

```
tensor<64x8xf32> → 32 个线程，每线程 4 个元素

         列0-3 (tid%2=0)           列4-7 (tid%2=1)
行0:  T0  [a₀₀ a₀₁ a₀₂ a₀₃]    T1  [a₀₄ a₀₅ a₀₆ a₀₇]
行1:  T2  [a₁₀ a₁₁ a₁₂ a₁₃]    T3  [a₁₄ a₁₅ a₁₆ a₁₇]
行2:  T4  [a₂₀ a₂₁ a₂₂ a₂₃]    T5  [a₂₄ a₂₅ a₂₆ a₂₇]
...
行15: T30 [a₁₅,₀ a₁₅,₁ a₁₅,₂ a₁₅,₃]   T31 [a₁₅,₄ a₁₅,₅ a₁₅,₆ a₁₅,₇]

(第一个 chunk: 行0-15, 然后 chunk2: 行16-31, chunk3: 行32-47, chunk4: 行48-63)
```

内存布局分析：

```
内存: [a₀₀ a₀₁ a₀₂ a₀₃ a₀₄ a₀₅ a₀₆ a₀₇ | a₁₀ a₁₁ ... a₁₇ | ...]
       ←—— T0 加载 (4 floats) ——→ ←—— T1 ——→

T0 和 T1 一起加载 8 个连续的 float = 32 bytes = 1 个 sector
一个 sector 正好覆盖 axis=1 方向的全部 8 个元素 (2 个线程各 4 个)
→ 访存完全合并
```

### 5.3 PTX 指令对比

**原始 kernel PTX：**

```ptx
.reqntid 256, 1, 1                                        // 256 线程 = 8 warps

// 标量加载
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];                // 1 个 float

// 5 轮 warp scan (32 线程, stride=1)
shfl.sync.up.b32    %r22, %r1, 1, 0, -1;                  // offset=1
shfl.sync.up.b32    %r24, %r23, 2, 0, -1;                 // offset=2
shfl.sync.up.b32    %r26, %r25, 4, 0, -1;                 // offset=4
shfl.sync.up.b32    %r28, %r27, 8, 0, -1;                 // offset=8
shfl.sync.up.b32    %r30, %r29, 16, 0, -1;                // offset=16

// 跨 warp 同步
@%p2 st.shared.b32 [ %r2 + 0 ], %r3;
bar.sync     0;
ld.shared.f32     %f17, [global_smem];

// 标量存储
@%p3 st.global.b32 [ %rd2 + 0 ], { %r4 };
```

**优化后 kernel PTX：**

```ptx
.reqntid 32                                                // 32 线程 = 1 warp

// 向量化加载 (128-bit = 4 x float)
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];     // 4 个 float
@%p2 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];     // chunk 2
@%p3 ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];  // chunk 3
@%p4 ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ]; // chunk 4

// 4 轮 warp scan (16 线程参与, stride=2)
shfl.sync.up.b32     %r50, %r1, 2, 0, -1;                 // offset=2 (threadStride=2)
add.f32     %r51, %r1, %r50;
selp.f32     %r52, %r1, %r51, %p10;

shfl.sync.up.b32     %r53, %r52, 4, 0, -1;                // offset=4
add.f32     %r54, %r52, %r53;
selp.f32     %r55, %r54, %r52, %p11;

shfl.sync.up.b32     %r56, %r55, 8, 0, -1;                // offset=8
add.f32     %r57, %r55, %r56;
selp.f32     %r58, %r57, %r55, %p12;

shfl.sync.up.b32     %r59, %r58, 16, 0, -1;               // offset=16
add.f32     %r60, %r58, %r59;
selp.f32     %r17, %r60, %r58, %p13;

// chunk 间传播 (单 warp, 无 shared memory)
shfl.sync.idx.b32     %r241, %r17, %r239, 31, -1;         // 获取上一个 chunk 的累加值
add.f32     %r21, %r105, %r241;                            // 累加

// 向量化存储
@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r17, %r18, %r19, %r20 };
@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r21, %r22, %r23, %r24 };
@%p3 st.global.v4.b32 [ %rd7 + 0 ], { %r25, %r26, %r27, %r28 };
@%p4 st.global.v4.b32 [ %rd8 + 0 ], { %r29, %r30, %r31, %r32 };
```

PTX 层面的差异对比：

| 方面 | 原始 kernel | 优化后 kernel |
|------|------------|--------------|
| 访存指令 | `ld/st.global.b32` (32-bit 标量) | `ld/st.global.v4.b32` (128-bit 向量) |
| 线程数 | 256 (8 warps) | 32 (1 warp) |
| Warp scan 轮数 | 5 轮 (log2(32)) | 4 轮 (log2(16)) |
| shfl offset 起始值 | 1 (threadStride=1) | 2 (threadStride=2) |
| 跨 warp 同步 | `st.shared` + `bar.sync` + `ld.shared` | 不需要 |
| chunk 间传播 | 通过 shared memory | `shfl.sync.idx`（寄存器级） |

### 5.4 为什么 shfl offset 从 2 开始

在优化后的 layout 中，`threadsPerWarp = [16, 2]`，线程在 warp 内的排列方式为：

```
laneId:  0  1  2  3  4  5  ... 30 31
axis=0:  0  0  1  1  2  2  ... 15 15
axis=1:  0  1  0  1  0  1  ...  0  1
```

即 lane 0 和 1 处理同一行的不同列，lane 2 和 3 处理下一行的不同列。因此 axis=0 方向相邻的两个线程间隔 2 个 lane，`threadStride = 2`。

Warp scan 需要沿 axis=0 传播，所以第一轮 `shfl.sync.up` 的 offset = 2（跳过同行的线程），后续分别为 4, 8, 16，共 4 轮（`log2(16) = 4`）。

### 5.5 chunk 间传播机制

`tensor<64x8>` 由 16 行的线程处理 64 行数据，需要 4 个 chunk（每个 chunk 16 行）。warp scan 完成后，每个 chunk 最后一行的线程持有该 chunk 的 inclusive sum。后续通过 `shfl.sync.idx` 获取前一个 chunk 的最后一个值，累加到当前 chunk 的所有元素上。

这对应编译器中的 `AddPartialReduceOneWarp` 路径——单 warp 时不需要 shared memory，直接 shuffle 传播。关于这个路径的实现细节，参见 [Triton Scan Op 编译全流程](https://zhuanlan.zhihu.com/p/2007221164263624909) 第五节。

## 六、vector cumsum：另一种 scan 实现

除了 scalar cumsum，代码中还有一个 `chunk_local_cumsum_vector_kernel`，用于 4D 输入 `[B, T, H, S]`。它使用了完全不同的算法——矩阵乘法实现 cumsum：

```python
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE", "HAS_SCALE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s, o, scale, cu_seqlens, chunk_indices, T,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
    REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    # ... varlen 处理 ...

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    # ... 构建 block_ptr ...
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    # ...
```

**思路**：构造一个下三角矩阵 `m_s`（大小为 `BT x BT`），然后通过 `tl.dot(m_s, b_s)` 完成 cumsum。

对于 `REVERSE=False`：

```
m_s = [[1, 0, 0, ...],
       [1, 1, 0, ...],
       [1, 1, 1, ...],
       ...]

cumsum(x) = m_s @ x
```

这种方法用 `O(BT^2 * BS)` 的计算换取更好的访存模式（可以利用 tensor core）。对于 vector 情况（S 维度较大），matmul 的吞吐量优势明显。

对比三种 kernel 的特点：

| 特性 | scalar (原始) | scalar (向量化) | vector (matmul) |
|------|--------------|----------------|-----------------|
| 输入维度 | [B, T, H] | [B, T, H] | [B, T, H, S] |
| 算法 | scan (shfl) | scan (shfl) | matmul (tl.dot) |
| 计算复杂度 | O(BT) | O(BT) | O(BT^2 * BS) |
| 访存模式 | 未合并 | 向量化合并 | tensor core 友好 |
| 适用场景 | - | scalar gate | vector gate |
| autotune | 无 | 无 | BS, num_warps |

## 七、优化效果

从 NCU profiling 结果确认：

| 指标 | 优化前 (scalar) | 优化后 (vectorization) |
|------|----------------|----------------------|
| Sectors/Req | 1（严重未合并） | 16（完全合并） |
| 向量化指令 | `ld/st.global.b32` | `ld/st.global.v4.b32` |
| 线程数 | 256（8 warps） | 32（1 warp） |
| Shared Memory | 使用（跨 warp 同步） | 不使用 |
| Warp scan 轮数 | 5 轮 | 4 轮 |
| 性能提升 (32k 序列) | baseline | 约 5 倍 |

长序列下性能提升更为显著。

## 八、总结

这个 case 里性能瓶颈不在算法，而在数据布局。原始 1D kernel 在 `[B, T, H]` 布局下做 T 维度 scan，stride = H 导致相邻线程读不同 cache line，`Sectors/Req = 1`。

修复方式是把 `tensor<BT>` 改为 `tensor<BT, BH>`，引入 H 维度的局部并行。Triton 编译器会自动推导出 `sizePerThread = [1, 4]` 的 blocked layout，生成 `ld/st.global.v4.b32`（128-bit 向量化指令）。附带好处是 autotune 选出单 warp 配置，不需要 shared memory 和 barrier。

几点体会：

- `make_block_ptr` 的形状和 stride 参数对最终生成的访存指令有直接影响。改 stride 从 `(H,)` 到 `(H, 1)` 就触发了完全不同的代码生成路径
- 对 bandwidth-bound 操作来说，单 warp + 向量化比多 warp + 标量更快。occupancy 低不是问题，跨 warp 同步才是
- 观察 TTGIR 的 blocked layout 参数是理解 Triton 代码生成的最直接方式。`sizePerThread` 决定向量化宽度，`threadsPerWarp` 决定 scan 并行度，`warpsPerCTA` 决定是否需要 shared memory
- NCU 的 `Sectors/Req` 指标能快速定位访存合并问题。正常的向量化 v4 加载应该是 16 sectors/req
