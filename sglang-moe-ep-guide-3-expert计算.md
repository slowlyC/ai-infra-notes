# SGLang DeepSeek MoE & Expert Parallelism 原理详解-2

## 6. Expert 计算 — Low-latency Mode

> **代码定位**：`DeepEPMoE.forward_impl()` → `run_moe_core()` → `forward_flashinfer_cutedsl()` → `flashinfer_cutedsl_moe_masked()`
> 文件：`layers/moe/ep_moe/layer.py`（路径选择），`layers/moe/flashinfer_cutedsl_moe.py`（`flashinfer_cutedsl_moe_masked()`），`layers/quantization/modelopt_quant.py`（权重定义）

本章覆盖 LL mode 下 MoE expert 计算的完整流程，包括 NVFP4 量化体系、各 kernel 的源码级分析、以及 layout/SMEM/TMEM 等硬件层面的实现细节。

### 6.1 NVFP4 量化体系

理解 expert 计算之前，需要先了解 NVFP4 的量化机制，因为每一步的输入输出 dtype、scale 参数都和这个体系有关。

#### 为什么需要两个 scale

FP4 (E2M1) 只有 4 bit，可表示的值仅 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` 及其负值，动态范围极窄。单靠一个 scale 很难同时兼顾整个 tensor 的全局范围和局部数值变化，因此 NVFP4 采用"两级 scale 相乘"的设计:

- **global scale** (per-tensor): 一个 float32 标量，覆盖整个 tensor 的数值范围，把 FP4 的有限表示映射到合理区间
- **block scale** (per-vector): 每 16 个连续元素一个 float8_e4m3 标量，捕捉局部数值的波动

反量化时两者相乘: `实际值 = FP4_raw × block_scale × (1 / global_scale)`。这相当于先用 global scale 做粗粒度的范围对齐，再用 block scale 做细粒度的局部修正。

#### 为什么 block scale 用 FP8

block scale 需要与 FP4 数据一起喂入 Tensor Core。Blackwell 的 `tcgen05.mma` 指令原生支持 "FP4 数据 + FP8 block scale" 的 mixed-input 模式: MMA 硬件在执行乘法时自动将 FP4 与对应的 FP8 scale 相乘，无需额外指令。如果 scale 用 FP16/FP32，就无法被硬件自动消费，需要在 GEMM 外部做手动反量化，性能会大幅退化。具体来说:

- `float8_e4m3fn` (E4M3): 4 bit 指数 + 3 bit 尾数，动态范围 ±240，精度适中，是 NVFP4 的默认 scale 类型
- `float8_e8m0fnu` (E8M0): 8 bit 指数无尾数，纯 power-of-2 scale，用于 MXFP4 格式

block scale 在存储时还需要做 swizzle 重排 (128x4 blockwise interleave)，匹配 Blackwell 硬件从 TMEM 读取 scale 的访问模式。

#### 三级量化粒度

NVFP4 的量化粒度从粗到细分三级:

**Per-tensor scale (global scale)**

每个 expert 一个 float32 标量，离线标定或从 checkpoint 加载。量化时所有元素共享同一个 scale。NVFP4 中 global scale 不直接乘到 FP4 值上，而是融入 GEMM kernel 的 epilogue alpha 参数。

对于权重: `alpha = 1 / weight_global_scale`，在 Gemm kernel 的 epilogue 中执行 `C = alpha * (A @ B^T)`。对于 activation: `input_global_scale` 在 FP4 量化时作为缩放因子，weight 和 activation 的 global scale 合并为 `alpha = weight_scale / input_scale`。

**Per-vector scale (block scale)**

每 `sf_vec_size` (=16) 个连续元素共享一个 `float8_e4m3` 标量，是 NVFP4 精度的主要保障。block scale 能捕捉到同一 tensor 内不同局部区域的数值范围差异，而 global scale 无法做到这一点。

block scale 的存储格式: swizzled (128x4 blockwise interleave)，由 `swizzle_blockscale` 函数完成重排。对于 K=7168 的维度，每 16 元素一个 scale，共 `ceil(7168/16)=448` 个 scale factor，swizzle 后尺寸 `ceil(K/64)*4 = 448`。

**Per-token scale (dynamic activation scale)**

lingjun 分支特有（`SGLANG_NVFP4_DYNAMIC_ACT_SCALE=1`），开源 SGLang 不包含此路径。前两级 scale 都是离线标定的静态值，而 per-token scale 在推理时实时计算: Gemm1 输出后、SiLU+FP4 量化前，对每个 expert 的有效行计算运行时 absmax，推导当前 batch 的 activation scale:

```
absmax = masked_absmax(gateup_output, masked_m)   → (L,) float32
a2_scale = (FP8_MAX × FP4_MAX) / absmax           # = 2688 / absmax
alpha2 = w2_weight_scale_2 / a2_scale
```

其中 `FP8_MAX=448, FP4_MAX=6`，乘积 2688 是 FP8×FP4 联合表示的理论最大值。这个公式的含义: 先算出把当前 activation 最大值映射到 FP8×FP4 满量程所需的 scale (`a2_scale`)，再结合权重 scale (`w2_weight_scale_2`) 得到 GEMM epilogue 使用的 `alpha2`。

相比 static scale（离线标定的固定值 `a2_global_scale`），dynamic scale 能自适应运行时 activation 的数值范围，减少 FP4 量化精度损失。代价是引入额外 kernel launch（`masked_absmax` 两阶段 reduce，约 18 us，详见 6.6 节）。


### 6.2 入口与 dispatch_output 解包

调用链：

```
DeepEPMoE.run_moe_core(dispatch_output)                   # layers/moe/ep_moe/layer.py
  │
  ├─DeepEPMoE.forward_flashinfer_cutedsl(dispatch_output) # layers/moe/ep_moe/layer.py
  │
  └─ quant_method.apply_without_routing_weights() # layers/quantization/modelopt_quant.py
       │
       └─ flashinfer_cutedsl_moe_masked(    # layers/moe/flashinfer_cutedsl_moe.py
              hidden_states = (data, scale),       # NVFP4: (uint8, 6D float8) | BF16: (bf16, None)
              input_global_scale,                  # (L,) float32 | None
              w1, w1_blockscale, w1_alpha,         # gate_up 权重 + block scale + alpha
              w2, a2_global_scale,                 # down 权重 + Gemm2 static act scale
              w2_blockscale, w2_alpha,             # down block scale + alpha
              w2_weight_scale_2,                   # w2 纯权重 scale (dynamic scale 时使用)
              masked_m,                            # (L,) int32, 每 expert 有效 token 数
              down_sm_count, down_signals, down_start_event  # 可选 overlap 参数
          )
```

forward_flashinfer_cutedsl 中对 dispatch_output 进行解包，得到 hidden_states，hidden_states_scale 和 masked_m。

```python
def forward_flashinfer_cutedsl(self, dispatch_output):
    hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
    # hidden_states.size():       (64, 512, 7168) bf16
    # hidden_states_scale.size(): (64, 512, 448) 或 None (BF16 dispatch 时)
    # masked_m.size():            (64,) int32, 每 expert 从全部 4 卡收到的 token, avg≈16
    output = self.quant_method.apply_without_routing_weights(
        layer=self, x=(hidden_states, hidden_states_scale), masked_m=masked_m, ...)
    return output
```

一路进入到flashinfer_cutedsl_moe.py的flashinfer_cutedsl_moe_masked()中。

#### hidden_states layout: 为什么 NVFP4 dispatch 是 (M, K/2, L) 而不是 (L, M, K)

上面写的 `(64, 512, 7168)` 是 BF16 dispatch 的格式。NVFP4 dispatch 时 hidden_states 的 shape 是 **(M, K/2, L)** (如 `(2048, 3584, 64)` uint8),expert 维度在最后。

这不是 sglang 做了 permute,而是 **DeepEP `low_latency_dispatch(use_nvfp4=True)` 直接返回的格式**。DeepEP 文档（`deep_ep/buffer.py` L689-692）:

> with `use_nvfp4=True`: the first element is a `torch.Tensor` shaped as `[num_max_dispatch_tokens_per_rank * num_ranks, hidden // 2, num_local_experts]` with `torch.uint8`. The second tensor is the corresponding scales for the first element with shape `[32, 4, num_max_dispatch_tokens_per_rank * num_ranks // 128, 4, hidden // 64, num_local_experts]` with `torch.float8_e4m3fn`.

对比不同 dispatch 模式的返回 layout:

| dispatch 模式 | hidden_states 格式 | expert 维度位置 |
|---|---|---|
| BF16 (`use_fp8=False, use_nvfp4=False`) | `(L, M, K)` bf16 | dim 0 (最前) |
| FP8 (`use_fp8=True`) | `(L, M, K)` fp8 + `(L, M, K//128)` scale | dim 0 (最前) |
| NVFP4 (`use_nvfp4=True`) | `(M, K/2, L)` uint8 + 6D scale | dim 2 (最后) |

NVFP4 模式之所以把 L 维放在最后,是因为 CuTeDSL `grouped_gemm_nt_masked` kernel 要求输入 A/B/C 的 layout 为 `(M, K, L)`,L 维在最外层 stride,以便 TMA (Tensor Memory Accelerator) 按 `(M, K)` tile 做地址计算。DeepEP 在 RDMA 通信阶段就把数据按 kernel 需要的 `(M, K/2, L)` layout 排好,省掉了一次 permute。

BF16 dispatch 时仍返回 `(L, M, K)`,进入 `flashinfer_cutedsl_moe_masked` 后由 `scaled_fp4_grouped_quantize` 内部完成从 `(L, M, K)` 到 `(M, K/2, L)` 的转换(详见 6.4 节)。

trace 中看到的 stride 验证了这个 layout:

```
aten::view  shape=(2048, 3584, 64)  stride=(3584, 1, 7340032)
```

stride[2]=7340032 = 2048×3584 最大,说明 L 维在最外层;stride[1]=1 说明 K/2 维最内层。整块内存物理上连续,逻辑 layout 是 `[L][M][K/2]` 的 column-major view。

### 6.3 执行步骤总览与数据流

源码：`layers/moe/flashinfer_cutedsl_moe.py` → `flashinfer_cutedsl_moe_masked()`。

以 decode batch=128, EP=4（无 DP attention）为例（trace 实测），完整的 tensor shape 变换和各 kernel 耗时：

```
hidden_states: (128, 7168) bf16
    │
    ├─ gate → topk → topk_ids: (128, 8)
    │
    ▼ DeepEP low_latency_dispatch (RDMA)                        ~13 us
    │  NVFP4 dispatch: (M, K/2, L) = (512, 3584, 64) uint8 + 6D scale
    │    DeepEP 直接返回 kernel 需要的 (M, K/2, L) layout, L 在最后
    │  BF16 dispatch:  (L, M, K) = (64, 512, 7168) bf16
hidden_states: (64, 512, 7168) bf16 或 (512, 3584, 64) uint8
masked_m: (64,) int32, avg ≈ 16
    │
    ├─ Step 1: FP4 输入量化 (仅 BF16 dispatch 时)               ~6 us      → 6.4 节
    │    scaled_fp4_grouped_quantize(hidden, masked_m, input_global_scale)
    │    → a_q: (64, 512, 3584) uint8, a_q_sf: (64, 512, 448) e4m3
    │    permute → (512, 3584, 64)                              (zero-copy, 见 6.4.1 节)
    │
    ├─ Step 2: Gemm1 — gate_up projection (CuTeDSL)            ~165 us    → 6.5 节
    │    grouped_gemm_nt_masked: (512,3584,64) FP4 × (4096,3584,64) FP4
    │    → gateup_output: (512, 4096, 64) bf16
    │
    ├─ Step 3: Dynamic scale (仅 lingjun 分支)                  ~18 us     → 6.6 节
    │    masked_absmax(gateup, masked_m) → per-expert absmax
    │    → a2_scale, alpha2: (64,) float32
    │
    ├─ Step 4: SiLU + FP4 量化                                  ~6 us      → 6.7 节
    │    silu_and_mul_scaled_nvfp4_experts_quantize(gateup, masked_m, a2_scale)
    │    → diq: (512, 1024, 64) uint8, diq_sf
    │
    ├─ Step 5: Gemm2 — down projection (CuTeDSL)               ~89 us     → 6.8 节
    │    grouped_gemm_nt_masked: (512,1024,64) FP4 × (7168,1024,64) FP4
    │    → out: (512, 7168, 64) bf16
    │
    ├─ permute back: (512, 7168, 64) → (64, 512, 7168)
    │
    ▼ DeepEP low_latency_combine (RDMA + topk_weights)          ~18 us
final: (128, 7168) bf16
```

**耗时分布（static scale, 单层）**：


| 阶段      | 耗时        | 占比  |
| ----------- | ------------- | ------- |
| dispatch  | 13 us       | 4.5%  |
| FP4 量化  | 6 us        | 2.1%  |
| Gemm1     | 165 us      | 57.3% |
| SiLU+量化 | 6 us        | 2.1%  |
| Gemm2     | 89 us       | 30.9% |
| combine   | 18 us       | 6.3%  |
| **总计**  | **~288 us** |       |

Dynamic scale 时 Gemm1 和 SiLU+量化之间插入 ~18 us 的 absmax + scale 推导，总计 ~306 us。

### 6.4 Step 1: `scaled_fp4_grouped_quantize` — 输入量化

> **源码定位**：
>
> - SGLang 调用端：`layers/moe/flashinfer_cutedsl_moe.py` L100-104
> - FlashInfer Python API：`flashinfer/fp4_quantization.py` → `scaled_fp4_grouped_quantize()`
> - 后端实现：`flashinfer/fp4_quantization.py` → `scaled_fp4_grouped_quant_sm100()`
> - C++ kernel：`flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp` → `silu_and_mul_scaled_nvfp4_experts_quantize()` with `use_silu_and_mul=False`

此步仅在 BF16 dispatch 时执行（`hidden_states[1] is None`），NVFP4 dispatch 时（`SGLANG_MOE_NVFP4_DISPATCH=1`）输入已量化，跳过。

**调用链**：

```
flashinfer_cutedsl_moe_masked()                    # SGLang
  │  hidden_states[1] is None → 需要量化
  │
  └─ flashinfer.scaled_fp4_grouped_quantize(       # flashinfer/fp4_quantization.py
         a = hidden_states[0],   # (64, 512, 7168) bf16
         mask = masked_m,        # (64,) int32
         a_global_sf = input_global_scale,) # (64,) float32, per-expert
       │
       └─ scaled_fp4_grouped_quant_sm100(input_tensor, input_global_scale, mask)
            │  l, m, k = 64, 512, 7168
            │  output = empty(64, 512, 3584) uint8        # 两个 FP4 packed 起来
            │  output_scales = empty(64, 512, 112) int32  # padded layout
            │
            └─ module.silu_and_mul_scaled_nvfp4_experts_quantize(
                   output, output_scales, input_tensor,
                   input_global_scale, mask,
                   False,) # use_silu_and_mul = False → 纯量化
```

FlashInfer API 内部（`flashinfer/fp4_quantization.py` → `scaled_fp4_grouped_quant_sm100` L490-546）负责分配输出 buffer、调用 C++ kernel、并执行 layout 变换。完整 Python 代码：

```python
# flashinfer/fp4_quantization.py — scaled_fp4_grouped_quant_sm100
def scaled_fp4_grouped_quant_sm100(
    input_tensor: torch.Tensor,      # (l, m, k) = (64, 512, 7168) bf16
    input_global_scale: torch.Tensor, # (l,) = (64,) float32
    mask: torch.Tensor,               # (l,) = (64,) int32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    output:        物理 (l, m, k//2), 逻辑 (m, k//2, l) — FP4 packed uint8
    output_scales: 物理 (l, rm, rk, 32, 4, 4), 逻辑 (32, 4, rm, 4, rk, l) — swizzled E4M3
    """
    device = input_tensor.device
    l, m, k = input_tensor.shape                         # 64, 512, 7168
    sf_vec_size = 16
    scale_k = k // sf_vec_size                           # 448
    padded_k = (scale_k + 3) // 4 * 4                   # 448 (已对齐)
    padded_k_int32 = padded_k // 4                       # 112
    padded_m = (m + 127) // 128 * 128                    # 512 (已对齐)

    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)       # (64, 512, 3584)
    output_scales = torch.empty(l, padded_m, padded_k_int32, device=device,
                                dtype=torch.int32)                              # (64, 512, 112)

    # 调用 C++ kernel (use_silu_and_mul=False → 纯量化，详见下方 C++ 代码)
    module.silu_and_mul_scaled_nvfp4_experts_quantize(
        output.view(l * m, k // 2),                        # flatten: (32768, 3584)
        output_scales.view(l * padded_m, padded_k_int32),  # flatten: (32768, 112)
        input_tensor.view(l * m, k),                       # flatten: (32768, 7168)
        input_global_scale, mask, False,
    )

    # ===== FP4 data permute =====
    # SGLang/DeepEP 约定 (L, M, K)，CuTeDSL grouped_gemm_nt_masked 要求 (M, K, L)。
    # 原因：CuTe 的 from_dlpack 把 stride 最大的维度视为 batch。
    # (64, 512, 3584) stride=(1835008, 3584, 1) → stride[0] 最大但在 dim0，CuTe 会合并 L×M。
    # permute 后 stride=(3584, 1, 1835008) → stride[2] 最大 → CuTe 识别 dim2=L 为 batch。
    # zero-copy：只改 stride 元数据，物理内存中 K 维始终连续 (stride=1)。
    output = output.permute(1, 2, 0)
    # (64, 512, 3584) → 逻辑 (512, 3584, 64)

    # ===== block scale swizzle =====
    # Blackwell tcgen05.mma 指令要求 block scale 按特定模式排列在 TMEM 中。
    # C++ kernel 已将 scale 写入 128×4 swizzled 物理布局 (l, padded_m, padded_k_int32)。
    # 这里做逻辑 reshape + permute，使 CuTeDSL 能正确解读 scale 的排列。
    #
    # 先 reinterpret int32 → 4 个 e4m3，再 reshape 为 6 维：
    #   128 行为一组 (padded_m//128)，组内 32×4=128 个 scale；K 维每 4 个为一组。
    output_scales = output_scales.view(torch.float8_e4m3fn)
    output_scales = output_scales.view(l, padded_m // 128, padded_k // 4, 32, 4, 4)
    #                                  (64,    4,           112,         32, 4, 4)
    # permute 成 (m32, m4, rm, k4, rk, l)，匹配 MMA 从 TMEM 读取 scale 的访问模式。
    # 同样 zero-copy。
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    #                                    (32, 4, 4, 4, 112, 64)
    return output, output_scales
```

SGLang 侧调用 `scaled_fp4_grouped_quantize()` 返回的 `a_q`、`a_q_sf` 已经是变换后的格式，可以直接传给 `grouped_gemm_nt_masked`。permute + swizzle 在后续所有量化输出处（6.4 纯量化、6.7 SiLU+量化）都会执行，代码完全相同。

整个 expert 计算中所有 permute 操作汇总：

```
Step 1 量化输出: a_q (64,512,3584) → permute(1,2,0) → (512,3584,64)     # FlashInfer API 内部做
Step 2 Gemm1 输出: (64,512,4096) → permute(1,2,0) → (512,4096,64)       # SGLang 分配后手动
Step 3 Gemm1 权重: w1 (64,4096,3584) → permute(1,2,0) → (4096,3584,64)
Step 4 SiLU+量化: gateup (512,4096,64) → permute(2,0,1) → (64,512,4096) # 量化 kernel 要求 expert-first
     输出 diq 内部 permute(1,2,0) → (512,1024,64)                         # 给 Gemm2
Step 5 Gemm2: w2 (64,7168,1024) → permute(1,2,0) → (7168,1024,64)
     out (64,512,7168) → permute(1,2,0) → (512,7168,64)
Step 6: out (512,7168,64) → permute(2,0,1) → (64,512,7168)               # 恢复 expert-first 给 combine
```

规律是 `permute(1,2,0)` 进 CuTeDSL，`permute(2,0,1)` 出来。Step 4 的来回 permute 是因为 CuTeDSL 和 FP4 量化 kernel（`cvt_fp16_to_fp4_expert`）的 layout 约定不同：CuTeDSL 要求 `(M,N,L)`，量化 kernel 要求 `(L,M,K)`。如果将 SiLU+量化迁移到 CuTeDSL 可以消除这次来回。

#### 6.4.1 C++ kernel 实现

上面 Python 代码中 `module.silu_and_mul_scaled_nvfp4_experts_quantize()` 调用的 C++ 实现位于 FlashInfer 内置的 NVIDIA 贡献代码中（`flashinfer/csrc/nv_internal/` 目录，保留了 `tensorrt_llm::kernels::` 命名空间，但 TRT-LLM 主仓库中没有这个 expert-masked 版本的量化 kernel）。

C++ 封装层（`fp4Quantize.cpp` L190-253）通过 `use_silu_and_mul` 标志区分模式，负责参数校验和 kernel 调度：

```cpp
// flashinfer/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp L190-253
void silu_and_mul_scaled_nvfp4_experts_quantize(
    Tensor output, Tensor output_scale, Tensor const input,
    Tensor const input_global_scale, Tensor const mask, bool use_silu_and_mul) {
  // ... 参数校验省略 ...
  constexpr int BLOCK_SIZE = 16;
  auto m_topk = input.shape()[0];    // l * m = 64 * 512 = 32768
  auto k_by_2 = input.shape()[1];    // 7168 (纯量化时) 或 4096 (SiLU+量化时)
  auto k = k_by_2;
  if (use_silu_and_mul) {
    k = k_by_2 / 2;                  // SiLU 模式: 前半 gate, 后半 up → k = 2048
  }
  auto n_experts = input_global_scale.shape()[0];  // 64
  // ...
  tensorrt_llm::kernels::invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>(
      output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
      input_global_scale.data_ptr(), mask.data_ptr(),
      use_silu_and_mul, m_topk, k, n_experts, stream);
}
```

实际 GPU kernel 是 `cvt_fp16_to_fp4_expert`（`quantization.cuh` L510-603）。每个 thread 处理 8 个元素（`CVT_FP4_ELTS_PER_THREAD=8`），grid 按 expert 数均匀分配 thread，mask 控制每个 expert 的有效行数：

```cpp
// flashinfer/csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh L510-603 — 简化
template <typename Type, bool UE8M0_SF>
__global__ __launch_bounds__(512, 4)
cvt_fp16_to_fp4_expert(int32_t numRows, int32_t numCols, Type const* in,
    float const* SFScale, uint32_t* out, uint32_t* SFout,
    int32_t* mask, bool use_silu_and_mul, int n_experts) {

  // 1. thread → expert 映射：将 grid 均分给 n_experts 个 expert
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;
  int expert_idx = tid / stride;
  int tid_in_expert = tid % stride;
  int m = numRows / n_experts;           // 每个 expert 的行数 (512)
  int padded_m = (m + 127) / 128 * 128;  // pad 到 128 的倍数
  int colsPerRow = numCols / 8;           // 每行多少个 8-element 向量

  // 2. 遍历当前 expert 的所有 (row, col) 位置
  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow;
       globalIdx < (expert_idx + 1) * m * colsPerRow;
       globalIdx += actual_stride) {
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;
    int rowIdx_in_expert = rowIdx - expert_idx * m;

    // 3. mask 提前退出：跳过 padding 行
    if (mask && rowIdx_in_expert >= mask[expert_idx]) break;

    // 4. 读 8 个 BF16 值 (向量化加载)
    PackedVecT in_vec = reinterpret_cast<PackedVecT const*>(in)[inOffset];

    // 5. 如果是 SiLU+量化模式，读后半部分并做 SiLU(gate) * up
    if (use_silu_and_mul) {
      PackedVecT in_vec_mul = reinterpret_cast<PackedVecT const*>(in)[inOffset + colsPerRow];
      silu_and_mul<Type, 8>(in_vec, in_vec_mul);
    }

    // 6. per-expert global scale
    float const SFScaleVal = SFScale[expert_idx];

    // 7. 计算 block scale 在 swizzled 输出中的位置
    uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<...>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 8. 8 个 BF16 → 4 个 FP4 pack + block scale (E4M3)
    out[outOffset] = cvt_warp_fp16_to_fp4<Type, 16, 8, false>(
        in_vec, SFScaleVal, sf_out);
  }
}
```

这个 kernel 的特点是 block scale 直接写入 swizzled 布局（通过 `cvt_quant_to_fp4_get_sf_out_offset` 计算偏移），因此 C++ 层输出的 scale 已经是 128×4 swizzled 格式，上面 Python 代码中的 reshape/permute 只是逻辑变换。

量化过程总结：

1. 读取 `mask[expert_id]`，只处理前 `mask[expert_id]` 行（跳过 padding）
2. 对每 16 个连续元素（`sf_vec_size=16`）计算 block-level absmax
3. 用 `input_global_scale[expert_id]` 和 block absmax 推导 block scale（E4M3 格式）
4. BF16 值除以 `(block_scale × input_global_scale)`，round 到 FP4（E2M1），两个 FP4 pack 进一个 uint8

### 6.5 Step 2: `grouped_gemm_nt_masked` — Gemm1 (gate_up projection)

> **源码定位**：
>
> - SGLang 调用端：`layers/moe/flashinfer_cutedsl_moe.py` L137-148
> - FlashInfer API：`flashinfer/cute_dsl/blockscaled_gemm.py` → `grouped_gemm_nt_masked()` L2947-3048
> - GPU kernel：同文件 → `Sm100BlockScaledPersistentDenseGemmKernel.kernel()` L976-1825

这是 LL mode 计算量最大的 kernel（trace ~165 us EP=4 / ~101 us EP=8），也是优化的首要目标。

#### 6.5.1 为什么用 "Dense" kernel 而不是 "Grouped" kernel

API 名叫 `grouped_gemm_nt_masked`，底层调用的是 `Sm100BlockScaledPersistentDenseGemmKernel`。这不是 bug，而是有意为之：

- **Grouped GEMM**：每个 group 的 M/N/K 可以不同，A/B/C 通过指针数组传入，kernel 需要在切换 group 时更新 TMA descriptor。group 数多（64）但每个 M 很小时开销明显。
- **Batched Dense GEMM**：所有 batch 共享相同的 M/N/K，A/B/C 是一个大 3D tensor。TMA descriptor 只创建一次，kernel 通过 L 维索引定位不同 expert。

DeepEP 的 LL mode 满足 batched dense 的前提：dispatch 后所有 expert 数据 pad 到统一 `(max_tokens, hidden_dim)` shape，放进 `(L, M, K)` 的 3D tensor。每个 expert 的有效行数由 `masked_m[i]` 指定，通过 `MaskedScheduler` 跳过 padding tile。64 个 expert 平均 `masked_m≈16`，每个 expert 只有 1 个 M-tile，如果用 Grouped GEMM 会频繁切换 group，batched dense + mask 方案更高效。

#### 6.5.2 API 调用

SGLang 侧在调用 Gemm1 前，先为输出 buffer 做 permute（`flashinfer_cutedsl_moe.py` L125-128）：

```python
gateup_output = torch.empty(
    (num_experts, m, n * 2), dtype=torch.bfloat16, device=a_q.device
)                                        # (64, 512, 4096) bf16
gateup_output = gateup_output.permute(1, 2, 0)  # → (512, 4096, 64)
```

这和 6.4.1 节 `a_q` 的 permute 原因相同：`grouped_gemm_nt_masked` 要求输出 tensor C 也是 `(M, N, L)` layout。SGLang 先分配 `(L, M, N)` 的 contiguous buffer（expert-first 是分配时的自然顺序），然后 `permute(1, 2, 0)` 得到 `(M, N, L)` 的逻辑 view，kernel 直接往这个 buffer 写入结果。这是 zero-copy 的。

```python
grouped_gemm_nt_masked(
    (a_q, a_q_sf),                                # lhs: A + SFA
    (w1.permute(1, 2, 0), w1_blockscale),         # rhs: B + SFB
    gateup_output,                                # out: C, 已 permute 为 (512, 4096, 64)
    masked_m,                                     # (64,) int32
    ab_dtype="float4_e2m1fn",                     # FP4 (E2M1)
    sf_dtype="float8_e4m3fn",                     # block scale 类型
    c_dtype="bfloat16",                           # 输出类型
    sf_vec_size=16,                               # 每 16 元素一个 block scale (NVF4)
    alpha=w1_alpha.view(1, 1, num_experts),       # (1, 1, 64), per-expert scale
    alpha_dtype="float32",
)
```

参数推导：`a_torch.shape = (512, 3584, 64)` → m=512, k=7168 (FP4 packed), l=64；`b_torch.shape = (4096, 3584, 64)` → n=4096。对 64 个 expert，每个做 `masked_m[i] × 4096 × 7168` 的 GEMM。kernel 按 `(m, n, k, l, dtypes, tile, cluster, sm_count)` 做编译缓存（`@functools.lru_cache`）。

#### 6.5.3 Kernel 架构：Warp 分工与 Persistent Loop

`Sm100BlockScaledPersistentDenseGemmKernel` 是 Blackwell (SM100) 专用的 persistent warp-specialized GEMM kernel，6 个 warp group（192 threads/CTA）：


| Warp ID | 角色           | 职责                                                              |
| --------- | ---------------- | ------------------------------------------------------------------- |
| 5       | TMA warp       | 从 GMEM 加载 A/B/SFA/SFB 到 SMEM，TMA prefetch descriptor         |
| 4       | MMA warp       | SFA/SFB 从 SMEM 拷贝到 TMEM，执行`tcgen05.mma` block-scaled MMA   |
| 0-3     | Epilogue warps | TMEM → Register → 类型转换 + alpha scaling → SMEM → TMA store |

三种 warp 角色各自运行独立的 persistent loop，通过 `MaskedScheduler` 协调 tile 分配：

```python
tile_sched = MaskedScheduler.create(tile_sched_params, block_idx(), grid_dim())
work_tile = tile_sched.initial_work_tile_info()
while work_tile.is_valid_tile:
    # ... 加载/计算/写出当前 tile
    tile_sched.advance_to_next_work()
    work_tile, _ = tile_sched.get_current_work()
```

`MaskedScheduler` 根据 `masked_m[batch_idx]` 计算每个 expert 需要多少个 M-tile，将这些 tile 按 `(M-tile, N-tile)` 二维编号为连续 `linear_idx`，每个 cluster 按步幅跳跃取 tile。当某个 expert 的 tile 取完后自动跳到下一个。padding 行不产生 tile。

以 Gemm1 为例：MMA tiler M=128，N-tile 数 = 4096/128 = 32。`masked_m[i]=18` → M-tile 数 = ceil(18/128) = 1，该 expert 1×32 = 32 个 tile。64 个 expert 总 tile 数约 2048 个。

#### 6.5.4 TMA Warp (Warp 5)

TMA warp 从 GMEM 加载数据到 SMEM，使用 Blackwell 的 TMA 硬件单元：

```python
# blockscaled_gemm.py L1262-1368 — 简化
if warp_idx == self.tma_warp_id:
    cpasync.prefetch_descriptor(tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb, tma_atom_c)
    while work_tile.is_valid_tile:
        for k_block in range(k_block_cnt):
            ab_pipeline.producer_acquire(ab_producer_state, ...)
            cute.copy(tma_atom_a, tAgA_slice[k], tAsA[stage], mcast_mask=...)   # A
            cute.copy(tma_atom_b, tBgB_slice[k], tBsB[stage], mcast_mask=...)   # B
            cute.copy(tma_atom_sfa, tAgSFA_slice[k], tAsSFA[stage], ...)         # SFA
            cute.copy(tma_atom_sfb, tBgSFB_slice[k], tBsSFB[stage], ...)         # SFB
            ab_producer_state.advance()
        tile_sched.advance_to_next_work()
```

K 维分 `k_block_cnt` 块。Gemm1：K=7168，`mma_tiler_k = 256`（FP4: 256bit/4bit=64 × inst_tile_k=4），`k_block_cnt = 7168/256 = 28`。TMA 支持 multicast：`cluster_shape_mn` 不为 `(1,1)` 时，A 沿 N 维 multicast，B 沿 M 维 multicast，减少 L2 流量。TMA 和 MMA 之间通过 `PipelineTmaUmma` 做 producer-consumer 同步，`num_ab_stage` 个 SMEM buffer。

#### 6.5.5 MMA Warp (Warp 4)

MMA warp 将 SFA/SFB 从 SMEM 拷贝到 TMEM（via `tcgen05.cp`），然后执行 `tcgen05.mma` block-scaled MMA：

```python
# blockscaled_gemm.py L1373-1572 — 简化
if warp_idx == self.mma_warp_id:
    while work_tile.is_valid_tile:
        acc_pipeline.producer_acquire(acc_producer_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in range(k_block_cnt):
            ab_pipeline.consumer_wait(ab_consumer_state, ...)
            cute.copy(tiled_copy_s2t_sfa, tCsSFA[stage], tCtSFA_s2t)  # SMEM → TMEM
            cute.copy(tiled_copy_s2t_sfb, tCsSFB[stage], tCtSFB_s2t)
            for kphase_idx in range(num_kphases):
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[kphase_idx].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[kphase_idx].iterator)
                cute.gemm(tiled_mma, tCtAcc, tCrA[kphase], tCrB[kphase], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
        acc_pipeline.producer_commit(acc_producer_state)
```

`tcgen05.mma.block_scale` 指令读 SMEM 中的 A/B 和 TMEM 中的 SFA/SFB，执行 `A × SFA × B × SFB` 并累加到 TMEM 中的 Float32 accumulator。block scale 在 MMA 指令级别原生支持。MMA 和 Epilogue 之间通过 `PipelineUmmaAsync` 同步。

#### 6.5.6 Epilogue Warps (Warp 0-3)

Epilogue warps 将 TMEM 中的 Float32 accumulator 转换为输出类型并写回 GMEM：

```python
# blockscaled_gemm.py L1576-1825 — 简化
if warp_idx < self.mma_warp_id:
    while work_tile.is_valid_tile:
        acc_pipeline.consumer_wait(acc_consumer_state)
        for subtile_idx in range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(subtile)], tTR_rAcc)   # TMEM → Register
            acc_vec = acc_vec * alpha[work_tile.tile_idx[2]]             # per-expert alpha
            acc_vec = acc_vec.to(self.c_dtype)                           # Float32 → BFloat16
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(buffer)])         # Register → SMEM
            cute.copy(tma_atom_c, bSG_sC[(buffer)], bSG_gC[(subtile)])  # TMA Store: SMEM → GMEM
```

alpha scaling 在 epilogue 应用：`alpha = w1_alpha[expert_id]`，是 NVFP4 两级量化中 `weight_global_scale / input_global_scale` 的合并缩放因子。

#### 6.5.7 `dst_signals` 机制

`dst_signals` 是可选的 `(num_experts,)` int32 tensor，用于 Gemm2 和 combine 的 overlap。Epilogue warp 在写完一个 expert 的所有 tile 后，通过 `atomic_add_release_global` 递增该 expert 的 signal counter。combine 端轮询 counter，一旦某 expert 的 signal 到达就立即开始该 expert 的 combine，不需要等所有 expert 算完。

`dsm_pending_packed`（u64，每 byte 代表一个 expert 的 pending count）和 `dsm_counter` 跟踪完成状态。Gemm1 不使用（传 None），只有 Gemm2 在 SBO combine overlap 模式下使用。

### 6.6 Step 3: `masked_absmax` — Dynamic Scale 计算

> **源码定位**：`layers/moe/flashinfer_cutedsl_moe.py` L26-155（lingjun 分支）
> **调用位置**：`flashinfer_cutedsl_moe_masked()` L287-294，在 Gemm1 输出和 SiLU+FP4 量化之间

此步仅在 `SGLANG_NVFP4_DYNAMIC_ACT_SCALE=1` 时执行（lingjun 分支），开源 SGLang 跳过。

Gemm1 输出是 3D padded tensor `(L, M, 2*intermediate)`，每个 expert 只有前 `masked_m[i]` 行有效。标准 `torch.abs().max()` 不支持 per-expert masked reduction，因此需要自定义 Triton kernel。

#### 6.6.1 两阶段 Reduce

```
输入: x (L, M, N), mask_m (L,)    # L=experts, M=max_buf, N=2*intermediate
  │
  ├─ Stage 1: absmax_partial_kernel
  │    grid: (L, ceil(M/32), ceil(N/2048))
  │    每个 thread block 处理一个 (expert, M-block, N-block) tile
  │    检查 mask_m[expert] 跳过无效行 → 对有效元素求 abs + max
  │    → partial_output: (L, num_m_blocks, num_n_blocks) float32
  │
  └─ Stage 2: absmax_final_reduce_kernel
       grid: (L,)
       每个 thread block 对一个 expert 的所有 partial max 求 final max
       → output: (L,) float32
```

Stage 1 的 tile 大小 `BLOCK_SIZE_M=32, BLOCK_SIZE_N=2048`。以 EP=8 `(32, 512, 4096)` 为例：grid = `(32, 16, 2)` = 1024 thread blocks，但大部分 M-block 的 `m_start >= masked_m[i]`（avg ~19 << 512）会 early return，实际有效 tile 约 `32 × 1 × 2 = 64` 个。

Stage 2 的 `BLOCK_SIZE_REDUCE = next_power_of_2(16 × 2) = 32`，grid = `(32,)`。

Stage 1 核心逻辑（简化）：

```python
@triton.jit
def absmax_partial_kernel(x_ptr, mask_ptr, partial_out_ptr, ...,
                          BLOCK_SIZE_M: tl.constexpr = 32,
                          BLOCK_SIZE_N: tl.constexpr = 2048):
    pid_b, pid_m, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_limit = tl.load(mask_ptr + pid_b)
    if pid_m * BLOCK_SIZE_M >= m_limit:
        tl.store(partial_out_ptr + ..., 0.0); return
    mask = (offs_m[:, None] < m_limit) & (offs_n[None, :] < N)
    vals = tl.load(ptr, mask=mask, other=0.0)
    tl.store(partial_out_ptr + ..., tl.max(tl.abs(vals.to(tl.float32))))
```

#### 6.6.2 Scale 推导

```python
if envs.SGLANG_NVFP4_DYNAMIC_ACT_SCALE.get():
    a2_scale = (448 * 6) / masked_absmax(gateup_output.permute(2, 0, 1), masked_m)
    alpha2 = w2_weight_scale_2 / a2_scale
else:
    a2_scale = a2_global_scale
    alpha2 = w2_alpha
```

`448 × 6 = 2688` 是 NVFP4 两级量化中 block scale × FP4 数值的理论最大乘积。

### 6.7 Step 4: `silu_and_mul_scaled_nvfp4_experts_quantize` — 激活 + 量化

> **源码定位**：
>
> - SGLang 调用端：`layers/moe/flashinfer_cutedsl_moe.py` L151-155
> - C++ kernel：`flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp` → `silu_and_mul_scaled_nvfp4_experts_quantize()` with `use_silu_and_mul=True`
> - CUDA kernel：`tensorrt_llm::kernels::invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>()`

这是 Gemm1 和 Gemm2 之间的融合 kernel：SiLU 激活 + element-wise multiply + NVFP4 量化，一次 kernel launch 完成。

**调用链**：

```
flashinfer_cutedsl_moe_masked()
  └─ silu_and_mul_scaled_nvfp4_experts_quantize(
         a = gateup_output.permute(2, 0, 1),  # (64, 512, 4096) bf16
         mask = masked_m,
         a_global_sf = a2_scale,)              # (64,) float32, static 或 dynamic
       └─ silu_and_mul_scaled_nvfp4_experts_quantize_sm100(input, mask, global_scale)
            │  l=64, m=512, k_by_2=4096 → k=2048
            │  output: (64, 512, 1024) uint8, output_scales: (64, 512, 32) int32
            └─ C++ kernel(output, output_scales, input, global_scale, mask, True)
```

**C++ kernel 内部逻辑**：输入 `(l×m, 2k) = (32768, 4096)`，前 2048 列是 gate，后 2048 列是 up。计算 `SiLU(gate) ⊙ up = (gate × sigmoid(gate)) ⊙ up`，然后对 activated 的每 16 个元素做 NVFP4 量化（block absmax → block scale E4M3 → round FP4 → pack uint8），mask 控制只处理每 expert 前 `masked_m[i]` 行。输出同样经过 permute + swizzle（同 6.4 节）。

与 `kernels.py` 中 Triton 实现的对比：`_silu_and_mul_post_quant_kernel` 做 SiLU + FP8 量化（DeepGemm 路径），`_silu_and_mul_post_per_tensor_quant_kernel` 做 SiLU + per-tensor FP8（W4AFP8 路径）。CuTeDSL 路径使用 FlashInfer 内置的 NVIDIA 贡献的 C++kernel（`cvt_fp16_to_fp4_expert`）做 SiLU+FP4 融合量化，因为 FP4 的 pack/swizzle 逻辑在 C++ 中实现更高效。

### 6.8 Step 5: `grouped_gemm_nt_masked` — Gemm2 (down projection)

> **源码定位**：同 6.5 节，共享 `Sm100BlockScaledPersistentDenseGemmKernel` kernel，仅参数不同。

#### 6.8.1 与 Gemm1 的参数差异

```python
grouped_gemm_nt_masked(
    (diq, diq_sf),                                   # A: (512, 1024, 64) FP4 packed
    (w2.permute(1, 2, 0), w2_blockscale),            # B: (7168, 1024, 64) FP4
    out,                                              # C: (512, 7168, 64) bf16
    masked_m,
    alpha=w2_alpha.view(1, 1, num_experts),           # (1, 1, 64), dynamic 时为 alpha2
    sm_count=down_sm_count,                           # 可选, 限制 SM 数
    dst_signals=down_signals,                         # 可选, signal combine 启动
)
```


| 维度              | Gemm1                      | Gemm2                 |
| ------------------- | ---------------------------- | ----------------------- |
| M (per expert)    | masked_m[i], avg ≈ 16     | 同左                  |
| N                 | 4096 (= 2 × intermediate) | 7168 (= hidden)       |
| K                 | 7168 (= hidden)            | 2048 (= intermediate) |
| K-block 数        | 7168/256 = 28              | 2048/256 = 8          |
| trace 耗时 (EP=4) | ~165 us                    | ~89 us                |

Gemm2 K 更小所以 K-loop 更短，耗时更少。N 更大（7168 vs 4096）但 K-loop 优势主导。

#### 6.8.2 `sm_count` — SM 数量限制

当 SBO combine overlap 启用时，`down_sm_count` 被设为比全部 SM 数小（如 120），剩余 SM 留给 combine 通信 kernel。Gemm2 的 grid 中 `max_active_clusters` 受 `sm_count` 约束。

#### 6.8.3 `dst_signals` — combine overlap

Gemm2 在 SBO combine overlap 模式下传入 `dst_signals`。epilogue 写完一个 expert 的所有输出 tile 后递增该 expert 的 signal，combine 端监听实现 expert 级别的 pipeline overlap。`down_gemm_overlap_args` 由 `ModelOptNvFp4FusedMoEMethod.apply_without_routing_weights` L1316-1333 提供。

**完整 MoE 层耗时（EP=8 LL Decode, 5742 层）**：


| 阶段                     | median (us) | 占比     |
| -------------------------- | ------------- | ---------- |
| Gemm1 (CuTeDSL)          | 101.06      | 55.6%    |
| Gemm2 (CuTeDSL)          | 53.12       | 29.2%    |
| Dynamic scale (5 kernel) | 18.02       | 9.9%     |
| SiLU + FP4 量化          | 9.79        | 5.4%     |
| **MoE 计算总计**         | **181.85**  | **100%** |
