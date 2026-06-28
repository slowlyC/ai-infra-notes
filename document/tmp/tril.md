## solve_tril 原理分析

### solve\_tril

该kernel计算下三角矩阵 (I + A)^-1 的逆，其中A是严格下三角矩阵（对角线为0）。

假设A为：

```shell
A = [0    0    0    0   ...]
    [a₁₀  0    0    0   ...]
    [a₂₀  a₂₁  0    0   ...]
    [a₃₀  a₃₁  a₃₂  0   ...]
    [ ...  ...  ...  ... ...]
```

则其对应的 (I + A) 矩阵：

```shell
I + A = [1    0    0    0   ...]
        [a₁₀  1    0    0   ...]
        [a₂₀  a₂₁  1    0   ...]
        [a₃₀  a₃₁  a₃₂  1   ...]
        [ ...  ...  ...  ... ...]
```

那么 (I + A) 的逆矩阵的第i行满足：

```shell
(I + A)^-1[i, j] = 
    -A[i, j] + Σ(k=0 to i-1) -A[i, k] * (I + A)^-1[k, j], if j < i
    1,                                                    if j = i  
    0,                                                    if j > i

```

### solve\_tril\_16x16\_kernel

计算16x16小块的下三角矩阵的逆

核心代码：

```shell
   for i in range(2, min(16, T - i_t * 16)):
        # [16]
        b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
```

### merge\_16x16\_to\_64x64\_inverse\_kernel

64x64矩阵可分为16个16x16块：

在计算时从对角线开始，逐块计算

```shell
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ A₁₁ (16x16) │     0       │      0      │      0      │  ← 行0-15
├─────────────┼─────────────┼─────────────┼─────────────┤
│ A₂₁ (16x16) │  A₂₂ (16x16)│      0      │      0      │  ← 行16-31
├─────────────┼─────────────┼─────────────┼─────────────┤
│ A₃₁ (16x16) │  A₃₂ (16x16)│  A₃₃ (16x16)│      0      │  ← 行32-47
├─────────────┼─────────────┼─────────────┼─────────────┤
│ A₄₁ (16x16) │  A₄₂ (16x16)│  A₄₃ (16x16)│  A₄₄ (16x16)│  ← 行48-63
└─────────────┴─────────────┴─────────────┴─────────────┘
```

即分块后下三角矩阵：

```shell
M = [A₁₁   0    0    0 ]
    [A₂₁  A₂₂   0    0 ]
    [A₃₁  A₃₂  A₃₃   0 ]
    [A₄₁  A₄₂  A₄₃  A₄₄]
```

由公式可得算法的计算流程图（数据依赖）

```shell
A₁₁⁻¹ ──┐
         ├─→ A₂₁⁻¹ = -A₂₂⁻¹A₂₁A₁₁⁻¹
A₂₂⁻¹ ──┘

A₂₂⁻¹ ──┐
         ├─→ A₃₂⁻¹ = -A₃₃⁻¹A₃₂A₂₂⁻¹
A₃₃⁻¹ ──┘

A₃₃⁻¹ ──┐
         ├─→ A₄₃⁻¹ = -A₄₄⁻¹A₄₃A₃₃⁻¹
A₄₄⁻¹ ──┘

A₁₁⁻¹ ──┐
A₂₁⁻¹ ──┼─→ A₃₁⁻¹ = -A₃₃⁻¹(A₃₁A₁₁⁻¹ + A₃₂A₂₁⁻¹)
A₃₃⁻¹ ──┘

A₂₂⁻¹ ──┐
A₃₂⁻¹ ──┼─→ A₄₂⁻¹ = -A₄₄⁻¹(A₄₂A₂₂⁻¹ + A₄₃A₃₂⁻¹)
A₄₄⁻¹ ──┘

A₁₁⁻¹ ──┐
A₂₁⁻¹ ──┼─→ A₄₁⁻¹ = -A₄₄⁻¹(A₄₁A₁₁⁻¹ + A₄₂A₂₁⁻¹ + A₄₃A₃₁⁻¹)
A₃₁⁻¹ ──┼─→
A₄₄⁻¹ ──┘
```

对应代码

```shell
    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION
    )

```

结论：可以看出在计算64x64矩阵的逆时，需要严格的数据依赖，且局部访存不佳。

因此使用not merge的方式，先并行计算对角线的16x16矩阵的逆，再计算大矩阵下三角矩阵的逆，可以有效提升并行度以及访存连续性。


## ncu分析
**结论：**当前开源实现的compute throughput和memory throughput都很低，只有20%左右，处于roofline的屋顶之下，计算流程存在问题；not merged后虽仍存在memory bound，但相比开源实现memory throughput已大幅增加。


not merge后单kernel性能提升明显，且序列长度越大越明显