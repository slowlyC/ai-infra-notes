## Triton-01-Vector Addition

### 前言

本系列教程记录了学习 Triton 的笔记，参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)，翻译并添加了一些自己的理解。

- **Triton-01-Vector Addition**（本文）
- Triton-02-Fused Softmax
- Triton-03-Matrix Multiplication
- Triton-04-Low-Memory Dropout
- Triton-05-Layer Normalization
- Triton-06-Fused Attention
- Triton-07-Extern Functions
- Triton-08-Grouped GEMM
- Triton-09-Persistent Matmul
- Triton-10-Block-Scaled Matmul
- Triton-11-Programmatic Dependent Launch

向量加法
===============

在本教程中，您将使用 Triton 编写一个简单的向量加法运算。

通过本教程，您将学习到: 

* Triton 的基本编程模型。

* `triton.jit` 装饰器，用于定义 Triton kernel。

* 针对自定义算子验证和基准测试的最佳实践，并与原生参考实现进行对比。

### 计算kernel

```python
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *指针* 指向第一个输入向量
               y_ptr,  # *指针* 指向第二个输入向量
               output_ptr,  # *指针* 指向输出向量
               n_elements,  # 向量的大小
               BLOCK_SIZE: tl.constexpr,  # 每个程序应该处理的元素数量
               # 注意: 使用 `constexpr` 以便它可以用作形状值
               ):
    # 有多个"程序"在处理不同的数据。我们在这里标识我们是哪个程序: 
    pid = tl.program_id(axis=0)  # 我们使用一维启动网格，所以轴是 0
    # 该程序将处理从初始数据block_start开始, 偏移offsets个元素。
    # 例如，如果你有一个长度为 256 的向量，块大小为 64，那么各个程序
    # 将分别访问元素 [0:64, 64:128, 128:192, 192:256]。
    # 注意 offsets 是一个指针列表。
    block_start = pid * BLOCK_SIZE  # 先定位block的起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建一个掩码来保护内存操作，防止越界访问
    mask = offsets < n_elements
    # 从 DRAM 加载 x 和 y，如果输入不是块大小的倍数，则屏蔽掉任何额外的元素
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回 DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

    ## make_block_ptr的形式
    # p_x = tl.make_block_ptr(base=x_ptr, shape=(n_elements,), strides=(1,), offsets=(pid * BLOCK_SIZE,), block_shape=(BLOCK_SIZE,), order=(0,))
    # p_y = tl.make_block_ptr(base=y_ptr, shape=(n_elements,), strides=(1,), offsets=(pid * BLOCK_SIZE,), block_shape=(BLOCK_SIZE,), order=(0,))
    # p_output = tl.make_block_ptr(base=output_ptr, shape=(n_elements,), strides=(1,), offsets=(pid * BLOCK_SIZE,), block_shape=(BLOCK_SIZE,), order=(0,))
    # x = tl.load(p_x, boundary_check=(0,))
    # y = tl.load(p_y, boundary_check=(0,))
    # output = x + y
    # tl.store(p_output, output, boundary_check=(0,))
```

让我们也声明一个辅助函数来 (1) 分配 `z` tensor
和 (2) 使用适当的网格/块大小将上述kernel加入队列: 

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # 我们需要预先分配输出tensor
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # SPMD 启动网格表示并行运行的kernel实例数量。
    # 它类似于 CUDA 启动网格。它可以是 Tuple[int] 或 Callable(metaparameters) -> Tuple[int]。
    # 在本例中，我们使用一维网格，其中大小是块的数量: 
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # 注意: 
    #  - 每个 torch.tensor 对象都会隐式转换为指向其第一个元素的指针。
    #  - `triton.jit` 装饰的函数可以用启动网格进行索引，以获得可调用的 GPU kernel。
    #  - 不要忘记将元参数作为关键字参数传递。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 我们返回 z 的句柄，但是由于还没有调用 `torch.cuda.synchronize()`，kernel此时
    # 仍在异步运行。
    return output
```

现在我们可以使用上面的函数来计算两个 `torch.tensor` 对象的逐元素和，并测试其正确性: 

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'torch 和 triton 之间的最大差异是 '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```


### 基准测试

现在我们可以在不断增大的向量上对我们的自定义算子进行基准测试，以了解它相对于 PyTorch 的表现如何。
为了简化操作，Triton 有一组内置实用工具，可以让我们简洁地绘制自定义算子的性能，
针对不同的问题规模。

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 用作绘图 x 轴的参数名称
        x_vals=[2**i for i in range(12, 28, 1)],  # `x_name` 的不同可能值
        x_log=True,  # x 轴是对数刻度
        line_arg='provider',  # 其值对应于图中不同线条的参数名称
        line_vals=['triton', 'torch'],  # `line_arg` 的可能值
        line_names=['Triton', 'Torch'],  # 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # 线条样式
        ylabel='GB/s',  # y 轴的标签名称
        plot_name='vector-add-performance',  # 绘图的名称。也用作保存绘图的文件名
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数的值
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
```

现在我们可以运行上面装饰的函数。传递 `print_data=True` 来查看性能数据，`show_plots=True` 来绘制它们，
和/或 `save_path='/path/to/results/'` 将它们与原始 CSV 数据一起保存到磁盘: 

```python
benchmark.run(print_data=True, show_plots=True)
```
