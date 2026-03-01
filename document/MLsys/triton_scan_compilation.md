# Triton Scan Op 编译全流程

在对 Qwen3.5/Qwen3-NEXT 模型中 GDN kernel 进行性能分析和优化的过程中，我们对 cumsum kernel 做了访存优化（参见 [cumsum 访存优化](https://zhuanlan.zhihu.com/p/2007222509582426207)）。 在这一过程中 autotune 结果显示 num_warps=1 的配置在大部分场景性能最优，出于对 Triton scan 实现的好奇，进一步追溯了编译器对 scan 算子的完整编译流程和 ir 实现。

本文从 `tl.cumsum` 的 Python 调用出发，拆解到最终 PTX 代码，覆盖 TTIR、TTGIR、LLVM IR、PTX 各阶段的 IR 变换，以及 scan 算法的底层实现（thread 内串行、warp 内 Kogge-Stone、跨 warp shared memory 协调）。

所用 Triton 版本：3.5.0+git50d10bdd


## 一、编译流程总览

```
Python层 (tl.cumsum / tl.associative_scan)
    ↓  ast_to_ttir (CodeGenerator, AST → MLIR)
Triton IR (TTIR)
    ↓  make_ttir (canonicalize, CSE, loop unroll, ...)
Triton IR (TTIR, optimized)
    ↓  make_ttgir (add layout encoding, coalesce, pipeline)
TritonGPU IR (TTGIR)
    ↓  make_llir (ScanOpToLLVM, emitFastScan)
LLVM IR (MLIR Dialect)
    ↓  llvm.to_module + llvm.optimize_module (O3)
LLVM IR (standard)
    ↓  llvm.translate_to_asm
PTX
    ↓  ptxas
CUBIN
```

源码中与 scan 相关的文件：

```
triton/
├── python/triton/
│   ├── language/
│   │   ├── standard.py              # cumsum 定义 → 调用 associative_scan
│   │   ├── core.py                  # associative_scan → 构建 scan region
│   │   └── semantic.py              # 调用 builder.create_scan
│   ├── compiler/
│   │   ├── compiler.py              # compile() 入口
│   │   └── code_generator.py        # AST → TTIR
│   └── runtime/jit.py               # @triton.jit 装饰器
├── python/src/ir.cc                  # pybind: create_scan → ScanOp
├── lib/
│   ├── Conversion/
│   │   ├── TritonToTritonGPU/
│   │   │   └── TritonToTritonGPUPass.cpp  # TritonScanPattern
│   │   └── TritonGPUToLLVM/
│   │       └── ScanOpToLLVM.cpp           # emitFastScan(关键)
│   └── Dialect/Triton/IR/Ops.cpp          # ScanOp verify
├── include/triton/Dialect/Triton/IR/
│   └── TritonOps.td                       # ScanOp TableGen 定义
└── third_party/nvidia/
    ├── backend/compiler.py                # NVIDIA make_ttir/ttgir/llir/ptx/cubin
    └── lib/TritonNVIDIAGPUToLLVM/
        └── TritonGPUToLLVM.cpp            # NVIDIA 特定 pattern 注册
```


## 二、Python 层：从 tl.cumsum 到 tt.scan

### 2.1 JIT 编译触发

`@triton.jit` 装饰器将 Python 函数包装为 `JITFunction`。调用 `kernel[grid](...)` 时触发编译：

```
JITFunction.__getitem__(grid)
  → LazyKernel.__call__(...)
    → JITFunction.run(...)
      → _do_compile(key, signature, device, ...)
```

`_do_compile` 的简化流程（`jit.py`）：

```python
def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
    kernel_cache, _, target, backend, _ = self.device_caches[device]
    src = self.ASTSource(self, signature, constexprs, attrs)
    kernel = self.compile(src, target=target, options=options.__dict__)
    kernel_cache[key] = kernel
    return kernel
```

### 2.2 compile 函数

`compiler.py` 中的 `compile()` 管理整个编译 pipeline：

```python
def compile(src, target=None, options=None, _env_vars=None):
    backend = make_backend(target)
    options = backend.parse_options(options)

    # 缓存检查
    key = get_cache_key(src, backend, options, env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    metadata_path = fn_cache_manager.get_file(metadata_filename)
    if metadata_path is not None:
        return CompiledKernel(src, metadata_group, hash)

    # 生成初始 TTIR
    context = ir.context()
    module = src.make_ir(target, options, codegen_fns, module_map, context)

    # 按 stages 编译: ttir → ttgir → llir → ptx → cubin
    stages = dict()
    backend.add_stages(stages, options, src.language)
    first_stage = list(stages.keys()).index(first_stage_name)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        module = next_module

    # 缓存结果
    fn_cache_manager.put(json.dumps(metadata_group), metadata_filename)
    return CompiledKernel(src, metadata_group, hash)
```

`backend.add_stages` 注册的 stages（NVIDIA backend）：

```python
def add_stages(self, stages, options, src_lang):
    stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, self.capability)
    stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
    stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
    stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)
```

编译过程中每个 stage 的产物都写入 `~/.triton/cache/<hash>/` 目录，可以直接查看各阶段的 IR。

### 2.3 cumsum 的 Python 调用链

```
tl.cumsum(input, axis=0, reverse=False, dtype=None)       # standard.py
  → core._promote_bfloat16_to_float32(input)              # bf16 提升为 fp32
  → _pick_sum_dtype(input.dtype, dtype)
  → core.associative_scan(input, axis, _sum_combine, reverse)  # core.py
    → semantic.associative_scan(input, axis, make_combine_region, reverse)
      → builder.create_scan(...)                           # ir.cc (pybind)
        → ScanOp 构造                                      # C++ MLIR Op
```

`standard.py` 中 cumsum 的定义：

```python
@core._tensor_member_fn
@jit
@core._add_scan_docstr("cumsum", dtype_arg="dtype")
def cumsum(input, axis=0, reverse=False, dtype: core.constexpr = None):
    input = core._promote_bfloat16_to_float32(input)
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)
    if out_dtype is not None:
        input = input.to(out_dtype)
    return core.associative_scan(input, axis, _sum_combine, reverse)
```

`_sum_combine` 就是 `lambda a, b: a + b`。`cumsum` 是 `associative_scan` 的特化，组合函数为加法。

bf16 输入会被提升为 fp32，因为 scan 的精度对误差累积敏感——每一步的输出都是下一步的输入，误差会沿 scan 方向传播放大。

### 2.4 associative_scan 的 Region 构建

`core.py` 中的 `associative_scan`：

```python
@_tensor_member_fn
@builtin
def associative_scan(input, axis, combine_fn, reverse=False,
                     _semantic=None, _generator=None):
    # 单 tensor 情况包装为 tuple
    if isinstance(input, tensor):
        return associative_scan((input, ), axis, combine_fn, reverse,
                               _semantic=_semantic, _generator=_generator)[0]

    def make_combine_region(scan_op):
        # 每个 tensor 对应两个标量参数: accumulator 和 current
        param_types = [t.type.scalar for t in input] * 2
        region = scan_op.get_region(0)
        builder = _semantic.builder

        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(
                region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty)
                    for i, ty in enumerate(param_types)]

            # 拆分为 left (accumulator) 和 right (current)
            n = len(input)
            left, right = args[:n], args[n:]

            # 调用组合函数 (cumsum → addf)
            results = _generator.call_JitFunction(
                combine_fn, left + right, kwargs={})

            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_scan_ret(*handles)

    axis = _unwrap_if_constexpr(axis)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    return _semantic.associative_scan(input, axis, make_combine_region, reverse)
```

`_semantic.associative_scan`（`semantic.py`）调用 C++ builder：

```python
def associative_scan(self, inputs, axis, region_builder_fn, reverse):
    shape = inputs[0].type.shape
    scan_op = self.builder.create_scan(
        [t.handle for t in inputs], axis, reverse)
    region_builder_fn(scan_op)
    assert scan_op.verify(), "scan op verification failed"
    return tuple(
        self.wrap_tensor(scan_op.get_result(i), inputs[i].type.scalar, shape)
        for i in range(len(inputs)))
```

`ir.cc` 中的 pybind：

```cpp
.def("create_scan",
     [](TritonOpBuilder &self, std::vector<Value> operands,
        int axis, bool reverse) -> OpState {
       return self.create<ScanOp>(operands, axis, reverse);
     })
```

Python 层到此为止。产出是一个 `tt.scan` MLIR 操作，带有输入 tensor、scan 方向（axis, reverse）和 combine region（cumsum 对应 `arith.addf`）。

### 2.5 ScanOp 的 TableGen 定义

`TritonOps.td` 中 ScanOp 的声明：

```tablegen
def TT_ScanOp : TT_Op<"scan", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    SingleBlockImplicitTerminator<"ScanReturnOp">
]> {
  let summary = "associative scan along an axis";
  let arguments = (ins
    Variadic<TT_Type>:$operands,
    I32Attr:$axis,
    BoolAttr:$reverse
  );
  let results = (outs Variadic<TT_Type>);
  let regions = (region SizedRegion<1>:$combineOp);
}
```

- `Variadic<TT_Type>:$operands`：支持多个输入（multi-value scan）
- `SizedRegion<1>:$combineOp`：恰好一个 region，包含组合逻辑
- `SingleBlockImplicitTerminator<"ScanReturnOp">`：region 以 `tt.scan.return` 结尾


## 三、TTIR 阶段

### 3.1 AST → TTIR

`CodeGenerator`（`code_generator.py`）继承自 `ast.NodeVisitor`，遍历 Python AST 生成 TTIR。对于 `tl.cumsum(b_s, axis=0)` 这样的调用，CodeGenerator 递归处理：

1. 解析 `tl.cumsum` → 找到 `standard.py` 中的 `cumsum` JIT 函数
2. 递归编译 `cumsum` → `associative_scan` → `semantic.associative_scan`
3. 通过 `builder.create_scan()` 在当前 module 中插入 `tt.scan` 操作

### 3.2 TTIR 优化 Pass

NVIDIA backend 的 `make_ttir`（`third_party/nvidia/backend/compiler.py`）：

```python
@staticmethod
def make_ttir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    if capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, 'make_ttir')
    return mod
```

各 pass 对 scan op 的影响：

| Pass | 作用 | 对 scan 的影响 |
|------|------|---------------|
| `add_inliner` | 内联 JIT 函数调用 | 将 cumsum/associative_scan 的调用内联到 kernel 中 |
| `add_rewrite_tensor_pointer` | 将 `tt.make_block_ptr` + `tt.load` 重写为 `tt.splat` + `tt.addptr` + `tt.load` | 影响 scan 的输入/输出加载和存储方式 |
| `add_canonicalizer` | 规范化：常量折叠、死操作消除等 | 简化 scan 前后的算术操作 |
| `add_combine` | 算术组合：`a + 0 → a`、`a * 1 → a` 等 | 清理冗余操作 |
| `add_cse` | 公共子表达式消除 | 合并重复计算 |
| `add_loop_unroll` | 循环展开 | scan 本身不涉及循环 |

`add_rewrite_tensor_pointer` 是一个重要 pass。在 TTIR 层，`tl.make_block_ptr` 会被降级为显式的指针算术（splat + addptr），展开后后续 pass 能看到具体的地址计算模式，coalesce pass 才能分析出访存是否连续。

### 3.3 TTIR 实例

以 FLA 代码中的 `chunk_local_cumsum_scalar_vectorization_kernel`（`HEAD_FIRST=False`, `BT=64`, `BH=8`）为例，经过 `make_ttir` 后的 IR：

```mlir
module {
  tt.func public @chunk_local_cumsum_scalar_vectorization_kernel(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg2: i32 {tt.max_val = 2147483647 : i32}) {

    // === 常量 ===
    %cst = arith.constant dense<32> : tensor<1x8xi64>       // H = 32
    %c8_i32 = arith.constant 8 : i32                        // BH
    %c64_i32 = arith.constant 64 : i32                      // BT
    %c32_i32 = arith.constant 32 : i32                      // H
    %c4_i32 = arith.constant 4 : i32                        // n_groups = cdiv(32, 8)

    // === program_id ===
    %0 = tt.get_program_id x : i32                           // i_t
    %1 = tt.get_program_id y : i32                           // i_bh

    // === 索引计算 ===
    %2 = arith.divsi %1, %c4_i32 : i32                      // i_b = i_bh / n_groups
    %3 = arith.remsi %1, %c4_i32 : i32                      // i_hg = i_bh % n_groups
    %4 = arith.muli %2, %arg2 : i32                         // i_b * T
    %5 = arith.muli %4, %c32_i32 : i32                      // bos * H = i_b * T * H
    %6 = arith.muli %0, %c64_i32 : i32                      // i_t * BT
    %7 = arith.muli %3, %c8_i32 : i32                       // i_hg * BH

    // === 指针构建 (rewrite_tensor_pointer 展开后) ===
    // 基地址
    %8 = tt.splat %arg0_ptr : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>

    // 行偏移: (i_t * BT + arange(0, 64)) * H
    %row_offsets = arith.muli %row_indices, %cst_H : tensor<64x1xi64>

    // 列偏移: i_hg * BH + arange(0, 8)
    %col_offsets = arith.extsi %col_indices : tensor<1x8xi32> to tensor<1x8xi64>

    // 总偏移 = base + row_offsets + col_offsets
    %total_offset = arith.addi %row_offsets_broadcast, %col_offsets_broadcast
    %ptrs = tt.addptr %8, %total_offset

    // === boundary check ===
    // row: (i_t * BT + arange(0, 64)) < T
    %row_mask = arith.cmpi slt, %row_indices, %T_splat : tensor<64x1xi32>
    // col: (i_hg * BH + arange(0, 8)) < H
    %col_mask = arith.cmpi slt, %col_indices, %H_splat : tensor<1x8xi32>
    // 合并 mask
    %mask = arith.andi %row_mask_broadcast, %col_mask_broadcast : tensor<64x8xi1>

    // === 加载 ===
    %b_s = tt.load %ptrs, %mask, %cst_zero
        : tensor<64x8x!tt.ptr<f32>>

    // === 类型转换 ===
    %b_s_f32 = arith.sitofp %b_s : tensor<64x8xi32> to tensor<64x8xf32>
    // (如果输入已经是 f32 则无此步)

    // === scan 算法===
    %b_o = "tt.scan"(%b_s_f32) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%acc: f32, %cur: f32):
      %sum = arith.addf %acc, %cur : f32
      tt.scan.return %sum : f32
    }) : (tensor<64x8xf32>) -> tensor<64x8xf32>

    // === 存储 ===
    tt.store %ptrs_o, %b_o, %mask : tensor<64x8x!tt.ptr<f32>>

    tt.return
  }
}
```

和 1D 的对比：

```mlir
// 原始 kernel (1D):
%b_o = "tt.scan"(%b_s) <{axis = 0, reverse = false}> ({
^bb0(%acc: f32, %cur: f32):
  %sum = arith.addf %acc, %cur : f32
  tt.scan.return %sum : f32
}) : (tensor<64xf32>) -> tensor<64xf32>

// 优化后 kernel (2D):
%b_o = "tt.scan"(%b_s) <{axis = 0, reverse = false}> ({
^bb0(%acc: f32, %cur: f32):
  %sum = arith.addf %acc, %cur : f32
  tt.scan.return %sum : f32
}) : (tensor<64x8xf32>) -> tensor<64x8xf32>
```

combine region 完全相同（都是标量 addf），差别仅在 tensor 形状。但 1D 和 2D 的形状差异会在 TTGIR 阶段推导出完全不同的 blocked layout，最终生成的 PTX 指令差别很大。


## 四、TTGIR 阶段：添加 Layout

### 4.1 TTIR → TTGIR 转换

`make_ttgir` 的 pass 序列：

```python
@staticmethod
def make_ttgir(mod, metadata, opt, capability):
    cluster_info = nvidia.ClusterInfo()
    if opt.cluster_dims is not None:
        cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ = opt.cluster_dims
    pm = ir.pass_manager(mod.context)

    # 添加 layout encoding
    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}",
                                       opt.num_warps, 32, opt.num_ctas)
    # Layout 优化
    passes.ttgpuir.add_coalesce(pm)
    if nvidia.has_matrix_core_feature(capability):
        passes.ttgpuir.add_accelerate_matmul(pm, capability)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    # 指令调度
    passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
    # 清理
    nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
    passes.ttgpuir.add_prefetch(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, True)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    if capability // 10 >= 9:
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(mod, 'make_ttgir')
    return mod
```

### 4.2 TritonScanPattern

`TritonToTritonGPUPass.cpp` 中将 TTIR 的 `tt.scan` 转换为 TTGIR 的 `tt.scan`（带 layout）：

```cpp
struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // adaptor.getOperands() 已经通过 TypeConverter 获得了带 layout 的类型
    auto newScan = rewriter.create<triton::ScanOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis(), op.getReverse());

    // 克隆 combine region（region 内是标量操作，不需要 layout）
    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());

    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};
```

`TritonGPUTypeConverter` 的工作是给每个 `RankedTensorType` 附加 `BlockedEncodingAttr`。layout 的具体参数由 `getDefaultBlockedEncoding` 计算，基于 `num_warps`、`threads_per_warp`、tensor shape 等信息推导。

### 4.3 Coalesce Pass

`add_coalesce` 会分析整个 module 的访存 pattern，尝试调整 layout 使得全局内存访问合并。对 scan op 来说，这个 pass 会检查 scan 的输入和输出的 load/store pattern，如果发现 layout 的 order 不利于合并访存，会插入 `ttg.convert_layout` 操作。

### 4.4 Blocked Layout 详解

#### 1D Layout（原始 kernel, num_warps=8）

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [8],
  order = [0]
}>
```

线程映射：

```
tensor<64xf32>:
  lane 0 → element 0     (warp 0)
  lane 1 → element 1     (warp 0)
  ...
  lane 31 → element 31   (warp 0)
  lane 0 → element 32    (warp 1)
  ...
  lane 31 → element 63   (warp 1)
  (warp 2-7 没有数据, 空闲)

总线程: 256, 有效线程: 64, 利用率: 25%
```

问题：
- 只有前 2 个 warp 有数据，6 个 warp 空闲
- 但 scan 需要跨 warp 协调所有 8 个 warp（shared memory + barrier）
- 浪费 SM 资源

#### 2D Layout（优化后 kernel, num_warps=1）

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1, 4],
  threadsPerWarp = [16, 2],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
```

线程映射的计算方式：

```
对于 tensor<64x8xf32>:

每个 warp "覆盖区域":
  axis=0: threadsPerWarp[0] × sizePerThread[0] = 16 × 1 = 16 行
  axis=1: threadsPerWarp[1] × sizePerThread[1] = 2 × 4 = 8 列

单个 warp 覆盖 16×8 = 128 个元素
tensor 有 64×8 = 512 个元素
需要 512/128 = 4 个 "scan blocks"（沿 axis=0 分 4 块）

laneId → (axis0_idx, axis1_idx):
  axis1_lane = laneId % threadsPerWarp[1] = laneId % 2
  axis0_lane = (laneId / threadsPerWarp[1]) % threadsPerWarp[0] = (laneId / 2) % 16

  axis0_idx = axis0_lane × sizePerThread[0] = axis0_lane
  axis1_idx = axis1_lane × sizePerThread[1] = axis1_lane × 4

具体映射:
  lane 0:  row=0,  col=0-3   (4 个连续 float)
  lane 1:  row=0,  col=4-7   (4 个连续 float)
  lane 2:  row=1,  col=0-3
  lane 3:  row=1,  col=4-7
  lane 4:  row=2,  col=0-3
  lane 5:  row=2,  col=4-7
  ...
  lane 30: row=15, col=0-3
  lane 31: row=15, col=4-7
```

即数据排布为:

```
        col 0-3             col 4-7
        (lane%2=0)          (lane%2=1)
row  0: lane 0 [*]          lane 1 [*]       <- 连续 8 个 float = 32 bytes
row  1: lane 2 [*]          lane 3 [*]
row  2: lane 4 [*]          lane 5 [*]
...
row 15: lane 30 [*]         lane 31 [*]
─────── scan block 0 boundary ───────
row 16: lane 0 [*]          lane 1 [*]       <- block 1 复用所有 lane
row 17: lane 2 [*]          lane 3 [*]
...
row 31: lane 30 [*]         lane 31 [*]
─────── scan block 1 boundary ───────
row 32-47: scan block 2
row 48-63: scan block 3
```

order = [1, 0] 的含义：axis=1 是"最快变化"的维度。lane 0 的 4 个元素 (col 0-3) 在内存中是连续的。lane 0 和 lane 1 的 8 个元素 (col 0-7) 也在同一行内连续。

warp 内相邻的 lane（lane 0 和 lane 1）加载的数据在内存中紧挨着，一次 128-byte 事务可以同时服务。

### 4.5 TTGIR 完整实例

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2],
                          warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1, "ttg.num-warps" = 1,
                   ttg.target = "cuda:90", "ttg.threads-per-warp" = 32} {

  tt.func public @chunk_local_cumsum_scalar_vectorization_kernel(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16},
      %arg2: i32) {

    // 所有 tensor 现在带有 #blocked encoding
    %ptrs = tt.splat %base : !tt.ptr<f32>
        -> tensor<64x8x!tt.ptr<f32>, #blocked>

    // ... 偏移计算 (同 TTIR，但所有 tensor 带 #blocked) ...

    %b_s = tt.load %ptrs, %mask, %zero
        : tensor<64x8x!tt.ptr<f32>, #blocked>

    // scan op 输入输出都带 layout
    %b_o = "tt.scan"(%b_s) <{axis = 0, reverse = false}> ({
    ^bb0(%acc: f32, %cur: f32):
      %sum = arith.addf %acc, %cur : f32
      tt.scan.return %sum : f32
    }) : (tensor<64x8xf32, #blocked>) -> tensor<64x8xf32, #blocked>

    tt.store %ptrs_o, %b_o, %mask
        : tensor<64x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
```


## 五、LLVM IR 阶段：Scan 的底层实现

这是 scan 编译中最复杂的阶段——将高层的 `tt.scan` 操作降级为具体的 shuffle、shared memory、barrier 等底层操作。

### 5.1 make_llir 的 Pass 序列

```python
def make_llir(self, src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)

    # 准备
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)

    # TritonGPU → LLVM (MLIR dialect)
    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)

    # 后处理
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)

    # NVGPU → LLVM
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
    passes.convert.add_nvvm_to_llvm(pm)

    pm.run(mod, 'make_llir')

    # MLIR → 标准 LLVM IR
    llvm.init_targets()
    context = llvm.context()
    llvm_mod = llvm.to_module(mod, context)

    # 目标设置
    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)
    features = get_features(options, self.target.arch)
    llvm.attach_datalayout(llvm_mod, triple, proc, features)

    # LLVM O3 优化
    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    metadata["shared"] = src.get_int_attr("ttg.shared")
    return str(llvm_mod)
```

### 5.2 ScanOp 转换注册

`TritonGPUToLLVM.cpp`（NVIDIA 特定）注册 scan pattern：

```cpp
struct ConvertTritonGPUToLLVM
    : public ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // ...
    RewritePatternSet patterns(context);

    // 注册各种 pattern（包括 scan）
    mlir::triton::populateScanOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, benefit);
    // ... load, store, dot, etc. ...

    if (applyPartialConversion(mod, target, std::move(patterns)).failed())
      signalPassFailure();
  }
};
```

`ScanOpToLLVM.cpp`：

```cpp
struct ScanOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp> {
  using ConvertTritonGPUReduceScanToLLVMPattern::ConvertTritonGPUReduceScanToLLVMPattern;

  LogicalResult matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (succeeded(emitFastScan(op, adaptor, rewriter, targetInfo)))
      return success();
    return failure();
  }
};

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
```

### 5.3 ScanLoweringHelper

在看 `emitFastScan` 之前，先理解辅助类 `ScanLoweringHelper`。它根据 layout 计算 scan 所需的各种参数：

```cpp
class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(ScanOp op) {
    auto type = cast<RankedTensorType>(op.getOperands()[0].getType());
    encoding = dyn_cast<BlockedEncodingAttr>(type.getEncoding());
    // ...
  }

  // 每个线程处理的元素数
  unsigned getAxisNumElementsPerThread() {
    return encoding.getSizePerThread()[axis];  // e.g. 1 for [1,4]
  }

  // warp 内的线程数
  unsigned getAxisNumThreadsPerWarp() {
    return encoding.getThreadsPerWarp()[axis];  // e.g. 16 for [16,2]
  }

  // 有唯一数据的线程数 (考虑 tensor 实际大小)
  unsigned getAxisNumThreadsPerWarpWithUniqueData() {
    return std::min(getAxisNumThreadsPerWarp(), shape[axis] / sizePerThread[axis]);
  }

  // warp 内线程的 stride (laneId 的步长)
  unsigned getAxisThreadStride() {
    unsigned stride = 1;
    for (int i = 0; i < axis; ++i)
      stride *= encoding.getThreadsPerWarp()[i] * encoding.getWarpsPerCTA()[i];
    // 对于 order=[1,0], axis=0:
    //   axis 0 的 order 是 1 (outer), axis 1 的 order 是 0 (inner)
    //   inner dims: axis 1, threadsPerWarp[1] = 2
    //   threadStride = 2
  }

  // element 的 stride
  unsigned getAxisElementStride() {
    // 对于 sizePerThread=[1,4], axis=0:
    // elementStride = product(sizePerThread[d] for d with higher order) = 4
  }

  // block 数量 (tensor axis size / per-warp coverage)
  unsigned getAxisNumBlocks() {
    return shape[axis] / (sizePerThread[axis] * threadsPerWarp[axis] * warpsPerCTA[axis]);
  }
};
```

以 `#blocked<{sizePerThread=[1,4], threadsPerWarp=[16,2], warpsPerCTA=[1,1], order=[1,0]}>` 和 `tensor<64x8>`, `axis=0` 为例：

| 参数 | 值 | 计算 |
|------|-----|------|
| `getAxisNumElementsPerThread()` | 1 | sizePerThread[0] = 1 |
| `getAxisNumThreadsPerWarp()` | 16 | threadsPerWarp[0] = 16 |
| `getAxisThreadStride()` | 2 | threadsPerWarp[1] = 2 (axis=1 是 inner dim) |
| `getAxisElementStride()` | 4 | sizePerThread[1] = 4 |
| `getAxisNumBlocks()` | 4 | 64 / (1 × 16 × 1) = 4 |
| `getNonAxisNumElementsPerThread()` | 4 | sizePerThread[1] = 4 |

### 5.4 emitFastScan 完整流程

`emitFastScan` 是 scan 降级的主要代码实现（`ScanOpToLLVM.cpp:463-565`）。以下是完整的逻辑：

```cpp
LogicalResult ScanOpConversion::emitFastScan(
    triton::ScanOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter,
    const TargetInfoBase &targetInfo) const {

  ScanLoweringHelper helper(op);
  auto loc = op.getLoc();
  auto mod = op->getParentOfType<ModuleOp>();

  // 获取线程标识
  Value threadId = getThreadId(rewriter, loc);
  unsigned iWarpSize = TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = b.i32_val(iWarpSize);
  Value warpId = b.udiv(threadId, warpSize);
  Value laneId = b.urem(threadId, warpSize);

  // 计算 scan axis 方向的 lane 和 warp 索引
  unsigned axisThreadStride = helper.getAxisThreadStride();
  unsigned axisWarpStride = helper.getAxisWarpStride();
  Value laneIdAxis = b.udiv(b.urem(laneId, b.i32_val(axisThreadStride *
                            helper.getAxisNumThreadsPerWarp())),
                            b.i32_val(axisThreadStride));
  Value warpIdAxis = b.udiv(b.urem(warpId, b.i32_val(axisWarpStride *
                            helper.getAxisNumWarps())),
                            b.i32_val(axisWarpStride));

  unsigned axisNumWarps = helper.getAxisNumWarps();

  // Unpack 输入: 从 LLVM struct 解包为 SmallVector<SmallVector<Value>>
  // srcValues[i][j] 表示第 i 个元素的第 j 个 operand
  auto srcValues = unpackInputs(loc, op, adaptor, rewriter, *getTypeConverter());

  // 处理 reverse: 翻转 scan 方向
  if (op.getReverse()) {
    warpIdAxis = b.sub(b.i32_val(axisNumWarps - 1), warpIdAxis);
    srcValues = flipSrcValues(loc, op, rewriter, targetInfo, srcValues, iWarpSize);
  }

  // ========== 三层 Scan ==========

  // 第1层: Thread 内 scan
  scanThreadContiguousElements(srcValues, rewriter, helper);

  // 第2层: Warp 内 scan (Kogge-Stone)
  warpScan(srcValues, rewriter, targetInfo, helper, laneIdAxis);

  // 第3层: 跨 Warp scan
  if (axisNumWarps > 1) {
    // 多 warp: shared memory 方式
    storeWarpAccumulator(srcValues, rewriter, helper,
                         laneId, warpId,
                         smemBases, smemTypes,
                         parallelLaneId, isRepresentative, targetInfo);
    b.barrier();
    AddPartialReduce(srcValues, rewriter, targetInfo, helper,
                     smemBases, smemTypes, warpId, laneIdAxis, parallelLaneId);
  } else if (srcValues.size() > 1) {
    // 单 warp, 多 scan block: 寄存器级传播
    AddPartialReduceOneWarp(srcValues, rewriter, targetInfo, helper,
                            laneIdAxis, laneId);
  }
  // else: 单 warp, 单 block → 不需要额外操作

  // 处理 reverse: 翻转回来
  if (op.getReverse()) {
    srcValues = flipSrcValues(loc, op, rewriter, targetInfo, srcValues, iWarpSize);
  }

  // Pack 结果
  SmallVector<Value> results(op.getNumOperands());
  auto valuesTransposed = transpose(srcValues);
  for (int i = 0; i < op.getNumOperands(); ++i) {
    results[i] = packLLElements(loc, getTypeConverter(),
                                valuesTransposed[i], rewriter, op.getResultTypes()[i]);
  }
  rewriter.replaceOp(op, results);
  return success();
}
```

### 5.5 第1层：Thread 内 Scan

`scanThreadContiguousElements`（`ScanOpToLLVM.cpp:28-48`）：

```cpp
static void scanThreadContiguousElements(
    SmallVector<SmallVector<Value>> &srcValues,
    ConversionPatternRewriter &rewriter,
    ScanLoweringHelper &helper) {

  unsigned scanElementsPerThread = helper.getAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned numChunks = helper.getAxisNumBlocks()
                     * helper.getNonAxisNumBlocks()
                     * helper.getNonAxisNumElementsPerThread();

  SmallVector<SmallVector<Value>> accs(numChunks);

  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    // cycleThrough: 根据 stride 计算当前元素属于哪个 chunk
    unsigned accIndex = cycleThrough(srcIndex, elementStride, scanElementsPerThread);
    // accumulate: 调用 combine region (addf)
    accs[accIndex] = accumulate(helper, rewriter, accs[accIndex], srcValues[srcIndex]);
    srcValues[srcIndex] = accs[accIndex];
  }
}
```

对于当前 layout（`sizePerThread[0] = 1`）：
- `scanElementsPerThread = 1`
- 每个线程只有 1 个元素
- 循环只执行 1 次，没有实质的 thread 内累加

如果是 `sizePerThread = [4, 1]`（每线程 4 个元素），则：
```
Thread 内串行累加:
  element[0] = a₀
  element[1] = a₀ + a₁
  element[2] = a₀ + a₁ + a₂
  element[3] = a₀ + a₁ + a₂ + a₃
```

### 5.6 第2层：Warp 内 Scan (Kogge-Stone)

Scan 算法的实现与 reduce 类似——都使用 warp 内 shuffle 指令进行树形归约，区别在于 scan 需要保留每个线程的中间结果（prefix），而 reduce 只需要最终的聚合值。关于 GPU reduce 的具体实现可以参考 [NVIDIA CUDA Reduce 优化详解](https://zhuanlan.zhihu.com/p/426978026)。

`warpScan`（`ScanOpToLLVM.cpp:52-83`）：

```cpp
static void warpScan(
    SmallVector<SmallVector<Value>> &srcValues,
    ConversionPatternRewriter &rewriter,
    const TargetInfoBase &targetInfo,
    ScanLoweringHelper &helper,
    Value laneIdAxis) {

  unsigned scanElementsPerThread = helper.getAxisNumElementsPerThread();
  unsigned elementStride = helper.getAxisElementStride();
  unsigned threadStride = helper.getAxisThreadStride();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();

  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    // 只对每个 scan block 的最后一个 per-thread 元素做 warp scan
    unsigned elementIdx = cycleThrough(srcIndex, elementStride, scanElementsPerThread);
    if (elementIdx != scanElementsPerThread - 1)
      continue;

    auto &acc = srcValues[srcIndex];

    // Kogge-Stone: log2(scanDim) 轮
    for (unsigned i = 1; i <= scanDim / 2; i <<= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned j = 0; j < acc.size(); ++j) {
        // shuffleUp: 从 lane - (i * threadStride) 获取数据
        shfl[j] = targetInfo.shuffleUp(rewriter, loc, acc[j],
                                       i * threadStride);
      }
      // mask: 只有 laneIdAxis >= i 的线程才执行累加
      Value mask = b.icmp_sge(laneIdAxis, b.i32_val(i));
      SmallVector<Value> tempAcc = accumulate(helper, rewriter,
                                              shfl, acc, mask);
      for (unsigned j = 0; j < acc.size(); ++j) {
        acc[j] = b.select(mask, tempAcc[j], acc[j]);
      }
    }
  }
}
```

以 `scanDim=16`, `threadStride=2` 为例：

```
循环: i = 1, 2, 4, 8

i=1: shfl_up offset = 1 × 2 = 2
     lane 0,1 → 保持不变 (laneIdAxis < 1)
     lane 2,3 → += lane 0,1 的值 (laneIdAxis >= 1)
     lane 4,5 → += lane 2,3 的值
     ...

i=2: shfl_up offset = 2 × 2 = 4
     lane 0-3 → 保持不变
     lane 4,5 → += lane 0,1
     lane 6,7 → += lane 2,3
     ...

i=4: shfl_up offset = 4 × 2 = 8
     lane 0-7 → 保持不变
     lane 8-15 → += lane 0-7
     ...

i=8: shfl_up offset = 8 × 2 = 16
     lane 0-15 → 保持不变
     lane 16-31 → += lane 0-15
     ...
```

4 轮之后每个 lane 持有正确的 inclusive prefix sum。

offset 是 `i * threadStride` 而非 `i`，因为 axis=0 相邻的线程在 warp 内不是相邻 lane。`threadStride=2` 说明 lane 0 和 lane 2 才是 axis=0 的"邻居"（lane 0 和 lane 1 是同一行的不同列）。shuffle 要跳 `2 × i` 个 lane 才对应 axis=0 方向的偏移 i。

### 5.7 第3层：跨 Warp Scan

#### 多 Warp (AddPartialReduce)

当 `axisNumWarps > 1` 时（如原始 kernel 的 8 warp 配置）：

```cpp
// Step 1: 存储每个 warp 的部分结果到 shared memory
static void storeWarpAccumulator(
    SmallVector<SmallVector<Value>> &srcValues, ...) {
  // 每个 warp 的最后一个 lane (代表线程) 存储 warp 的累加值
  // shared memory 布局: [numWarps × numChunks × numOperands]
  Value isRepresentative = /* laneIdAxis == scanDim - 1 */;
  // if (isRepresentative) smem[warpId][chunk][operand] = accValue;
}

// Step 2: barrier
b.barrier();  // → llvm.nvvm.barrier0

// Step 3: 读取并累加
static void AddPartialReduce(
    SmallVector<SmallVector<Value>> &srcValues, ...) {
  // 每个线程读取比自己 warpId 小的所有 warp 的部分结果
  // 串行累加（warp 数量通常很少，4-8 个）
  for (unsigned i = 0; i < axisNumWarps; ++i) {
    Value smemValue = load(smem[i][chunk][operand]);
    Value shouldAccumulate = icmp_slt(i, warpIdAxis);
    acc = select(shouldAccumulate, combine(acc, smemValue), acc);
  }
  // 将 acc 加到当前 warp 的所有元素上
}
```

对应的 PTX（8 warp 配置）：

```ptx
// 存储到 shared memory
@%p_representative st.shared.b32 [ global_smem + %offset ], %r_acc;

// barrier
bar.sync 0;

// 读取
ld.shared.f32 %f_w0, [global_smem + 0];     // warp 0 的累加
ld.shared.f32 %f_w1, [global_smem + 4];     // warp 1 的累加
// ...

// 根据当前 warpId 选择性累加
setp.gt.u32 %p_w0, %r_warpId, 0;
@%p_w0 add.f32 %f_acc, %f_acc, %f_w0;
setp.gt.u32 %p_w1, %r_warpId, 1;
@%p_w1 add.f32 %f_acc, %f_acc, %f_w1;
// ...
```

#### 单 Warp, 多 Block (AddPartialReduceOneWarp)

当只有 1 个 warp 但 tensor 的 scan 维度超过 warp 覆盖范围时（如 64 行 / 16 行 per block = 4 blocks）：

```cpp
static void AddPartialReduceOneWarp(
    SmallVector<SmallVector<Value>> &srcValues,
    ConversionPatternRewriter &rewriter,
    const TargetInfoBase &targetInfo,
    ScanLoweringHelper &helper,
    Value laneIdAxis, Value laneId) {

  unsigned scanElementsPerThread = helper.getAxisNumElementsPerThread();
  unsigned scanDim = helper.getAxisNumThreadsPerWarpWithUniqueData();
  unsigned numScanBlocks = helper.getAxisNumBlocks();
  unsigned threadStride = helper.getAxisThreadStride();

  SmallVector<Value> accumulator;
  unsigned axisBlockId = 0;

  for (unsigned srcIndex = 0; srcIndex < srcValues.size(); srcIndex++) {
    unsigned elementIdx = cycleThrough(srcIndex, ...);
    if (elementIdx != scanElementsPerThread - 1)
      continue;

    if (axisBlockId == 0) {
      // 第一个 block: 直接用 warp scan 结果作为 accumulator
      accumulator = srcValues[srcIndex];
    } else {
      // 后续 block: 累加前一个 block 的最后值
      srcValues[srcIndex] = accumulate(helper, rewriter,
                                       accumulator, srcValues[srcIndex]);
    }

    // 获取当前 block 的最后一个值，传播到下一个 block
    auto lastElement = srcValues[srcIndex];
    if (scanDim > 1) {
      for (unsigned i = 0; i < helper.getNumOperands(); ++i) {
        // shuffleUp: 把 accumulator 传给 block 内所有非首 lane
        lastElement[i] = targetInfo.shuffleUp(rewriter, loc,
            srcValues[srcIndex][i], threadStride);
        lastElement[i] = b.select(maskFirstLane, accumulator[i], lastElement[i]);

        if (numScanBlocks > 1) {
          // shuffleIdx: 获取最后一个 lane 的值作为下一个 block 的 prefix
          accumulator[i] = targetInfo.shuffleIdx(rewriter, loc,
              srcValues[srcIndex][i], laneIdLast);
        }
      }
    }

    // 更新 block 内其他元素 (sizePerThread > 1 的情况)
    for (unsigned i = 1; i < scanElementsPerThread; ++i) {
      auto &laneValue = srcValues[srcIndex - i * elementStride];
      laneValue = accumulate(helper, rewriter, lastElement, laneValue);
    }

    axisBlockId++;
  }
}
```

对应 PTX：

```ptx
// block 0 的 warp scan 结果在 %r17 (每个线程的值)

// 获取 block 0 最后一个 lane 的值
and.b32     %r238, %r_tid, 1;                    // axis=1 的 lane 部分
or.b32     %r239, %r238, 30;                     // laneIdLast = (tid & 1) | 30
shfl.sync.idx.b32  %r241, %r17, %r239, 31, -1;  // 获取 lane 30/31 的值

// block 1 的每个元素 += block 0 的总和
add.f32     %r105_new, %r105, %r241;
// ...

// 获取 block 1 最后一个 lane 的值
shfl.sync.idx.b32  %r242, %r105_new, %r239, 31, -1;

// block 2 的每个元素 += block 0+1 的总和
add.f32     %r193_new, %r193, %r242;
// ...
```

`laneIdLast` 的计算：对于 `threadsPerWarp = [16, 2]`，axis=0 最后一个线程的 lane 是 `30`（lane 30: axis0=15, axis1=0）或 `31`（lane 31: axis0=15, axis1=1）。每个 axis=1 的线程需要找到自己对应的"最后一行"线程，所以 `laneIdLast = (laneId & 1) | 30`。

### 5.8 生成的 LLVM IR 片段

```llvm
define ptx_kernel void @chunk_local_cumsum_scalar_vectorization_kernel(
    ptr addrspace(1) %arg0,
    ptr addrspace(1) %arg1,
    i32 %arg2) local_unnamed_addr #0 {

  ; === 线程 ID ===
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %2 = and i32 %1, 31                                       ; laneId
  %3 = lshr i32 %2, 1                                       ; laneIdAxis0 = laneId / 2

  ; === 地址计算 (省略) ===

  ; === 向量化加载 (ld.global.v4.b32) ===
  %58 = tail call { i32, i32, i32, i32 } asm sideeffect
    "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09@$5 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];",
    "=r,=r,=r,=r,l,b"(ptr addrspace(1) %45, i1 %54) #5
  %59 = extractvalue { i32, i32, i32, i32 } %58, 0
  %60 = extractvalue { i32, i32, i32, i32 } %58, 1
  %61 = extractvalue { i32, i32, i32, i32 } %58, 2
  %62 = extractvalue { i32, i32, i32, i32 } %58, 3

  ; (对 4 个 scan block 各做一次加载, 共 16 个 i32 值)

  ; === bitcast i32 → float (PTX 用 i32 存 float) ===
  %63 = bitcast i32 %59 to float
  %64 = bitcast i32 %60 to float
  %65 = bitcast i32 %61 to float
  %66 = bitcast i32 %62 to float

  ; === Warp Scan 步骤1: shfl.sync.up offset=2 ===
  %94 = tail call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %59, i32 2, i32 0)
  %95 = bitcast i32 %94 to float
  %.not = icmp samesign ult i32 %3, 1              ; laneIdAxis < 1 ?
  %96 = fadd float %63, %95
  %97 = select i1 %.not, float %63, float %96      ; selp

  ; === Warp Scan 步骤2: shfl.sync.up offset=4 ===
  %98 = bitcast float %97 to i32
  %99 = tail call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %98, i32 4, i32 0)
  %100 = bitcast i32 %99 to float
  %.not1 = icmp samesign ult i32 %3, 2              ; laneIdAxis < 2 ?
  %101 = fadd float %97, %100
  %102 = select i1 %.not1, float %97, float %101

  ; === Warp Scan 步骤3: shfl.sync.up offset=8 ===
  %104 = bitcast float %102 to i32
  %105 = tail call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %104, i32 8, i32 0)
  %106 = bitcast i32 %105 to float
  %.not2 = icmp samesign ult i32 %3, 4
  %107 = fadd float %102, %106
  %108 = select i1 %.not2, float %102, float %107

  ; === Warp Scan 步骤4: shfl.sync.up offset=16 ===
  %110 = bitcast float %108 to i32
  %111 = tail call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %110, i32 16, i32 0)
  %112 = bitcast i32 %111 to float
  %.not3 = icmp samesign ult i32 %3, 8
  %113 = fadd float %108, %112
  %114 = select i1 %.not3, float %108, float %113   ; warp scan 完成

  ; (对 %r2, %r3, %r4 重复相同模式)

  ; === Block 间传播 (shfl.sync.idx) ===
  %238 = and i32 %1, 1
  %239 = or disjoint i32 %238, 30                    ; laneIdLast
  %240 = bitcast float %114 to i32
  %241 = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %240, i32 %239, i32 31)
  %242 = bitcast i32 %241 to float
  ; block 1 += block 0 最后值
  %243 = fadd float %block1_val, %242

  ; === 向量化存储 ===
  tail call void asm sideeffect
    "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };",
    "r,r,r,r,l,b"(i32 %403, i32 %407, i32 %411, i32 %415, ptr addrspace(1) %475, i1 %54) #5

  ret void
}
```


## 六、PTX 与 CUBIN

### 6.1 PTX 生成

```python
def make_ptx(self, src, metadata, opt, capability):
    ptx_version = get_ptx_version_from_options(opt, self.target.arch)
    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)
    features = get_features(opt, self.target.arch)
    flags = ["nvptx-mad-wide-opt"]

    ret = llvm.translate_to_asm(src, triple, proc, features, flags,
                                opt.enable_fp_fusion, False)

    # 提取 kernel 名称
    names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
    metadata["name"] = names[0]

    # 修正 PTX 版本和目标
    ptx_version_str = f'{ptx_version//10}.{ptx_version%10}'
    ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version_str}', ret)
    ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret)
    return ret
```

### 6.2 完整 PTX 输出

优化后 kernel 的 PTX（简化版，保留关键指令）：

```ptx
.version 9.0
.target sm_90a
.address_size 64

.visible .entry chunk_local_cumsum_scalar_vectorization_kernel(
    .param .u64 .ptr .global .align 16 param_0,    // s
    .param .u64 .ptr .global .align 16 param_1,    // o
    .param .u32 param_2                             // T
)
.reqntid 32
{
    .reg .pred   %p<14>;
    .reg .b32    %r<272>;
    .reg .b64    %rd<33>;

    // =========== 地址计算 ===========
    ld.param.u64     %rd1, [param_0];              // s ptr
    ld.param.u64     %rd2, [param_1];              // o ptr
    ld.param.u32     %r1, [param_2];               // T

    mov.u32     %r33, %ctaid.x;                    // i_t (chunk index)
    mov.u32     %r35, %ctaid.y;                    // i_bh (batch×head_group)
    mov.u32     %r41, %tid.x;                      // threadIdx.x
    and.b32     %r42, %r41, 31;                    // laneId = tid & 31
    shr.u32     %r43, %r42, 1;                     // laneIdAxis0 = laneId >> 1

    // i_b, i_hg 计算
    div.u32     %r36, %r35, 4;                     // i_b = i_bh / n_groups
    rem.u32     %r37, %r35, 4;                     // i_hg = i_bh % n_groups

    // 基地址: s + bos * H
    mul.lo.s32     %r44, %r36, %r1;                // i_b * T
    mul.lo.s32     %r45, %r44, 32;                 // bos * H

    // 行偏移: (laneIdAxis0 + i_t * BT) * H
    shl.b32     %r46, %r33, 6;                     // i_t * 64 (i_t * BT)
    add.s32     %r47, %r43, %r46;                  // laneIdAxis0 + i_t * BT
    mul.lo.s32     %r48, %r47, 32;                 // * H

    // 列偏移: laneId%2 * 4 + i_hg * BH
    and.b32     %r49, %r42, 1;                     // laneId & 1
    shl.b32     %r50, %r49, 2;                     // * 4 (sizePerThread[1])
    shl.b32     %r51, %r37, 3;                     // i_hg * 8 (i_hg * BH)
    add.s32     %r52, %r50, %r51;

    // 总偏移
    add.s32     %r53, %r48, %r52;
    add.s32     %r54, %r45, %r53;

    // boundary check
    setp.lt.s32     %p1, %r47, %r1;                // row < T ?
    setp.lt.s32     %p2, %r52, 32;                 // col < H ?
    and.pred     %p3, %p1, %p2;                    // 合并 mask

    // =========== 向量化加载 (4 个 block) ===========
    // block 0: row 0-15
    mul.wide.s32     %rd10, %r54, 4;               // byte offset
    add.s64     %rd11, %rd1, %rd10;
    @%p3 ld.global.v4.b32 {%r101, %r102, %r103, %r104}, [%rd11];

    // block 1: row 16-31 (偏移 += 16 * H = 16 * 32 = 512 floats)
    add.s64     %rd12, %rd11, 2048;                // 512 * 4 bytes
    @%p3 ld.global.v4.b32 {%r105, %r106, %r107, %r108}, [%rd12];

    // block 2: row 32-47
    add.s64     %rd13, %rd12, 2048;
    @%p3 ld.global.v4.b32 {%r109, %r110, %r111, %r112}, [%rd13];

    // block 3: row 48-63
    add.s64     %rd14, %rd13, 2048;
    @%p3 ld.global.v4.b32 {%r113, %r114, %r115, %r116}, [%rd14];

    // =========== Warp Scan (对 %r101 示例) ===========
    // 步骤1: offset=2
    shfl.sync.up.b32     %r120, %r101, 2, 0, -1;
    setp.lt.u32     %p5, %r43, 1;                  // laneIdAxis0 < 1
    mov.b32     %f1, %r101;
    mov.b32     %f2, %r120;
    add.f32     %f3, %f1, %f2;
    selp.f32     %f4, %f1, %f3, %p5;               // mask: 保持或累加

    // 步骤2: offset=4
    mov.b32     %r121, %f4;
    shfl.sync.up.b32     %r122, %r121, 4, 0, -1;
    setp.gt.u32     %p6, %r43, 1;                  // laneIdAxis0 > 1
    mov.b32     %f5, %r122;
    add.f32     %f6, %f4, %f5;
    selp.f32     %f7, %f6, %f4, %p6;

    // 步骤3: offset=8
    mov.b32     %r123, %f7;
    shfl.sync.up.b32     %r124, %r123, 8, 0, -1;
    setp.gt.u32     %p7, %r43, 3;
    mov.b32     %f8, %r124;
    add.f32     %f9, %f7, %f8;
    selp.f32     %f10, %f9, %f7, %p7;

    // 步骤4: offset=16
    mov.b32     %r125, %f10;
    shfl.sync.up.b32     %r126, %r125, 16, 0, -1;
    setp.gt.u32     %p8, %r43, 7;
    mov.b32     %f11, %r126;
    add.f32     %f12, %f10, %f11;
    selp.f32     %f13, %f12, %f10, %p8;
    mov.b32     %r201, %f13;                        // warp scan 完成

    // (对 %r102, %r103, %r104, %r105-r116 重复)

    // =========== Block 间传播 ===========
    // 获取 laneIdLast
    and.b32     %r238, %r41, 1;
    or.b32     %r239, %r238, 30;                    // laneIdLast = (tid & 1) | 30

    // block 0 → block 1
    shfl.sync.idx.b32     %r241, %r201, %r239, 31, -1;
    mov.b32     %f20, %r241;
    mov.b32     %f21, %r205;                        // block 1 当前值
    add.f32     %f22, %f21, %f20;                   // += block 0 sum
    mov.b32     %r205, %f22;

    // block 1 → block 2
    shfl.sync.idx.b32     %r242, %r205, %r239, 31, -1;
    // ...

    // block 2 → block 3
    shfl.sync.idx.b32     %r243, %r209, %r239, 31, -1;
    // ...

    // =========== 向量化存储 ===========
    @%p3 st.global.v4.b32 [%rd21], {%r201, %r202, %r203, %r204};
    @%p3 st.global.v4.b32 [%rd22], {%r205, %r206, %r207, %r208};
    @%p3 st.global.v4.b32 [%rd23], {%r209, %r210, %r211, %r212};
    @%p3 st.global.v4.b32 [%rd24], {%r213, %r214, %r215, %r216};

    ret;
}
```

PTX 指令速查：

| 指令 | 用途 |
|------|------|
| `shfl.sync.up.b32 dst, src, offset, 0, -1` | 从 lane-(offset) 获取数据。mask=0xFFFFFFFF (-1) 表示全 warp 参与 |
| `shfl.sync.idx.b32 dst, src, idx, 31, -1` | 从指定 lane 获取数据。clamp=31 限制最大 lane |
| `ld.global.v4.b32 {r0,r1,r2,r3}, [addr]` | 128-bit 全局内存加载，4 个 32-bit 寄存器 |
| `st.global.v4.b32 [addr], {r0,r1,r2,r3}` | 128-bit 全局内存存储 |
| `selp.f32 dst, src1, src2, pred` | `dst = pred ? src1 : src2`，无分支条件选择 |
| `setp.lt.u32 pred, a, b` | `pred = (a < b)`，设置谓词寄存器 |

### 6.3 CUBIN 生成

```python
def make_cubin(self, src, metadata, opt, capability):
    ptxas = get_ptxas(self.target.arch).path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ptx') as fsrc:
        fsrc.write(src)
        fsrc.flush()
        fbin = fsrc.name + '.o'
        flog = fsrc.name + '.log'

        arch = get_sm_arch(capability)  # e.g. "sm_90a"
        ptxas_cmd = [
            ptxas, '-lineinfo', '-v',
            f'--gpu-name={arch}',
            fsrc.name, '-o', fbin
        ]
        try:
            subprocess.run(ptxas_cmd, check=True, stderr=open(flog, 'w'))
        except subprocess.CalledProcessError as e:
            with open(flog) as f:
                log = f.read()
            raise RuntimeError(f"ptxas failed:\n{log}")

        with open(fbin, 'rb') as f:
            cubin = f.read()
    return cubin
```

`ptxas` 输出的 verbose 信息中包含寄存器使用量和 shared memory 大小，这些信息会影响 occupancy：

```
ptxas info: Used 32 registers, 0 bytes smem, 384 bytes cmem[0]
```


## 七、NCU 性能分析

### 7.1 Memory

**Shared Memory**：未使用。单 warp 配置走 `AddPartialReduceOneWarp`，不需要 shared memory。

**L1 Cache**：Hit Rate 约 1%。cumsum 的访存模式是 streaming（每个元素 load 一次 → compute → store 一次），没有数据复用。L1 cache line 128 bytes，但每次 v4 加载只读 16 bytes，cache line 剩余部分被浪费。

**L2 Cache**：Hit Rate 约 71%。原因是 load 和 store 访问相同地址模式——load 把数据拉入 L2，store 写回时 L2 命中。

**Sectors/Req**：优化后为 16。一次 `ld.global.v4.b32` 请求 16 bytes（128-bit），一个 warp 32 个线程同时请求，合计 512 bytes。512 / 32 = 16 个 sector（每个 sector 32 bytes）。这意味着访存完全合并。

### 7.2 Occupancy

单 warp 的理论 occupancy 低（1/N，N 为 SM 上可调度的最大 warp 数）。但 cumsum 是带宽受限操作，occupancy 不是瓶颈。

### 7.3 Warp States

主要 stall 原因是 `wait`——`shfl.sync.up` 的结果是下一条指令的输入，指令级并行度受限。这是 scan 数据依赖导致的固有限制。

分支发散接近零——所有线程执行相同的指令序列，通过 `selp` 谓词选择结果，不存在 warp divergence。


## 总结

Triton 的 scan 算子采用三层结构：thread 内串行累加、warp 内 Kogge-Stone shuffle、跨 warp shared memory 协调。编译器在 TTGIR 阶段根据 layout encoding 确定每层的参数（sizePerThread、threadsPerWarp、warpsPerCTA），在 LLVM IR lowering 阶段将三层结构展开为具体的 shuffle、shared memory 和 barrier 指令。

| 选择 | 原因 |
|------|------|
| 单 warp + 多 scan block | 避免 shared memory 和 barrier；用 shfl.idx 在 block 间传播 |
| sizePerThread=[1,4] | 让编译器生成 v4.b32 向量化指令 |
| threadsPerWarp=[16,2] | scan axis 16 线程（4 轮 shfl）; 非 scan axis 2 线程覆盖 8 列 |
| Kogge-Stone | O(N log N) 工作量，log N 步完成，warp-level 天然适配 |
| selp 替代分支 | 消除分支发散 |
| order=[1,0] | 列优先遍历，保证同行数据在内存中连续 |

回到最初的问题：autotune 显示 num_warps=1 最优，原因是 cumsum kernel 的数据量极小（64 个标量），多 warp 引入了不必要的 shared memory 同步和空闲 warp 开销。单 warp 配置下 scan 退化为纯 shuffle 操作（4 轮 Kogge-Stone），避免了跨 warp 协调的代价。

这次分析也帮助我们更好地理解了 Triton 编译器的 layout 系统如何影响 scan 的指令生成——2D tensor 的 blocked layout 让编译器自动推导出向量化访存，这正是 [cumsum 访存优化](https://zhuanlan.zhihu.com/p/2007222509582426207) 中性能提升的根本原因。
