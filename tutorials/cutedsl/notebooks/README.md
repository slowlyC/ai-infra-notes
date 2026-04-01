# CuTe DSL 中文教程（精简版）

本目录包含 CuTe DSL 的精简版中文教程，将原始的 12 个 notebook 文件合并整理为 5 个，方便系统学习。

## 教程目录

### 1. [cutedsl-01-basics.ipynb](./cutedsl-01-basics.ipynb) - CuTe DSL 基础教程

涵盖 CuTe DSL 的基础知识：
- **Hello World**：如何编写在 GPU 上运行的简单程序
- **数据类型**：CuTe DSL 中的原始数值类型（Int8/16/32/64, Float16/32/64 等）
- **Tensor**：CuTe 的核心数据结构，包括 Engine + Layout
- **TensorSSA**：静态单赋值形式的 tensor 值，用于寄存器级操作
- **打印调试**：在 CuTe 中打印静态和动态值的不同方式

---

### 2. [cutedsl-02-layout_algebra.ipynb](./cutedsl-02-layout_algebra.ipynb) - CuTe DSL 布局代数

深入理解 CuTe 的布局系统：
- **Coalesce**：布局合并操作
- **Composition**：布局组合操作
- **Division**：布局划分操作（logical_divide, zipped_divide, tiled_divide, flat_divide）
- **Product**：布局乘积操作（logical_product, blocked_product, raked_product）
- **组合布局**：使用 inner/offset/outer 实现复杂数据变换
- **Gather/Scatter**：使用组合布局实现间接访问模式

---

### 3. [cutedsl-03-elementwise_kernel.ipynb](./cutedsl-03-elementwise_kernel.ipynb) - 逐元素 Kernel 实现

学习如何构建高效的 GPU kernel：
- **朴素实现**：基本的逐元素加法 kernel
- **向量化**：使用向量化加载/存储提升性能
- **TV 布局**：Thread & Value 布局的概念和使用
- **高级优化**：thread block 重映射、自定义操作等
- **Benchmark**：使用 benchmark 工具测量性能
- **Autotune**：使用 autotune 工具自动调优参数

---

### 4. [cutedsl-04-gemm.ipynb](./cutedsl-04-gemm.ipynb) - GEMM 极限性能实现

基于 Blackwell (tcgen05) 的高性能 GEMM 实现：
- **GEMM 基础**：理解通用矩阵乘法的原理
- **Kernel 结构**：Prologue、Mainloop、Epilogue
- **软件流水线**：使用多阶段流水线隐藏 GMEM 延迟
- **累加器子分块**：优化寄存器使用
- **向量化存储**：向量化输出指令

---

### 5. [cutedsl-05-advanced_cuda.ipynb](./cutedsl-05-advanced_cuda.ipynb) - CuTe DSL 高级 CUDA 技术

高级 CUDA 编程技术：
- **Warp 间通信**：理解 CUDA Warp 和 Shared Memory
- **异步流水线**：使用 PipelineAsync 实现生产者-消费者模式
- **多级流水线**：带循环缓冲的分级异步流水线
- **CUDA Graphs**：减少 kernel 启动开销

---

## 学习路线

建议按顺序学习这些教程：

```
cutedsl-01-basics              基础概念
        ↓
cutedsl-02-layout_algebra      布局系统
        ↓
cutedsl-03-elementwise_kernel  简单 kernel 实现
        ↓
cutedsl-04-gemm                高性能 GEMM
        ↓
cutedsl-05-advanced_cuda       高级技术
```

## 前置要求

- Python 3.8+
- CUDA 12.0+
- PyTorch 2.0+
- NVIDIA CUTLASS

## 相关资源

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CuTe C++ 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
