"""
程序化依赖启动
=====================
本脚本演示了在向量加法示例的基础上使用 Triton 的程序化依赖启动(PDL)。

有关程序化依赖启动的 CUDA 参考，请参见 https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization。
有关程序化依赖启动的 PTX 参考，请参见 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol。

PDL (Programmatic Dependent Launch) 是 NVIDIA 在 Hopper (H100, 计算能力 9.0+) 及更新架构
中引入的硬件特性, 允许 GPU 上运行的 kernel 直接启动依赖 kernel, 无需返回 CPU 调度。

使用 PDL 后:
- CPU 启动 kernel A (启用 PDL)
- kernel A 在 GPU 上运行
- kernel A 直接在 GPU 上触发 kernel B (无需 CPU 参与)
- kernel B 开始执行


传统模式 (无 PDL):
时间线───────────────────────────────────────────────────>

CPU:  [启动kernelA] ──等待──> [同步] ──> [启动kernelB] ──等待──> [同步]
         │                    ↑            │                ↑
         ↓                    │            ↓                │
GPU:    [执行kernelA] ─────────┘           [执行kernelB] ──────┘
         └─写入GPU内存                    └─读取GPU内存

-----------------------------------------------------------------------

PDL 模式:
时间线───────────────────────────────────────────────────>

CPU:  [启动kernelA(PDL)] ──────────────────> [等待所有完成]
         │
         ↓
GPU:    [执行kernelA] ──gdc_launch──> [执行kernelB]
         │                           │
         └─写入GPU内存               └─读取GPU内存
         
         [kernel A 直接触发 kernel B, 无需 CPU 参与]

.. code-block:: bash
    python 11-programmatic-dependent-launch.py
"""

import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_pdl():
    """检查设备是否支持程序化依赖启动"""
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# 在此示例中
@triton.jit
def add_kernel(x_ptr,  #
               y_ptr,  #
               output_ptr,  #
               n_elements,  #
               BLOCK_SIZE: tl.constexpr,  #
               USE_GDC: tl.constexpr,  #
               ):
    # 获取程序 ID 并计算偏移量
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if USE_GDC:
        # GDC 等待会等待前一个kernel中的所有程序完成后再继续。
        # 这确保任何内存操作在wait指令之前按程序顺序发生，
        # 例如，如果前一个kernel写入 x 或 y，新值将可见。
        tl.extra.cuda.gdc_wait()

    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    if USE_GDC:
        # GDC 启动依赖项提示运行时系统启动依赖kernel。(通知硬件: "我准备好启动下一个kernel了")
        # 这些依赖kernel也必须在启用 PDL 的情况下启动。
        # 一旦所有程序发出 GDC 启动或程序已完成，如果有足够的资源，依赖网格就可以开始。
        # 注意: 这本身不提供额外的内存顺序保证，与 `gdc_wait` 不同
        tl.extra.cuda.gdc_launch_dependents()
    
    # 执行向量加法
    output = x + y
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

# 假设有三个kernel按顺序执行: 
# Kernel A: 数据预处理
# Kernel B: 向量加法（本示例）
# Kernel C: 结果后处理

# 完整数据流: 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# [Kernel A - 10个Programs并行]
#   读取原始数据
#   处理...
#   gdc_launch_dependents()
#   写入中间结果到 GPU 内存
#             │
#             ↓ 所有Program都调用了gdc_launch
# [硬件自动启动 Kernel B]
#             ↓
# [Kernel B - 10个Programs并行]
#   gdc_wait()                ← 等待 Kernel A 完全完成, 确保内存写入可见
#   读取 x, y
#   计算 output = x + y
#   gdc_launch_dependents()
#   写入结果到 GPU 内存
#             │
#             ↓ 所有Program都调用了gdc_launch
# [硬件自动启动 Kernel C]
#             ↓
# [Kernel C - Programs并行]
#   gdc_wait()                ← 等待 Kernel B 完全完成
#   读取 Kernel B 的输出
#   后处理...
#   写入最终结果

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 整个过程中，CPU 只参与了最初的启动！

def add(x: torch.Tensor, y: torch.Tensor, launch_pdl: bool = True):
    """
    使用 Triton 执行向量加法
    
    参数:
        x: 第一个输入tensor
        y: 第二个输入tensor
        launch_pdl: 是否使用程序化依赖启动
    
    返回:
        输出tensor(x + y)
    """
    output = torch.empty_like(x)
    assert x.device == y.device and output.device == x.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=1024,
        USE_GDC=launch_pdl,  # 在kernel中设置 constexpr 以使用网格依赖控制
        launch_pdl=launch_pdl,  # 在启用 PDL 标志的情况下启动kernel
    )
    return output


def validate(n_elements):
    """验证向量加法的正确性"""
    x = torch.rand(n_elements, device="cuda", dtype=torch.float32)
    y = torch.rand(n_elements, device="cuda", dtype=torch.float32)

    # 使用 PyTorch 计算参考结果
    torch_result = x + y
    # 使用 Triton 计算结果
    add_result = add(x, y)

    # 比较结果
    torch_vs_add = "✅" if torch.allclose(torch_result, add_result, atol=1.0) else "❌"
    print(f"元素数量={n_elements} 验证朴素方法 vs: ", end="")
    print(f"加法: {torch_vs_add}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # 用作图表 x 轴的参数名称
        x_vals=[2**i for i in range(23, 28, 1)],  # `x_name` 的不同可能值
        x_log=False,  # x 轴是对数的
        line_arg="provider",  # 其值对应于图表中不同行的参数名称
        line_vals=["pdl-fp32", "fp32"],  # `line_arg` 的可能值
        line_names=["PDL", "No PDL"],  # 行的标签名称
        styles=[("red", "-"), ("blue", "-")],  # 行样式
        ylabel='GB/s',  # y 轴的标签名称
        plot_name="pdl-performance",  # 图表的名称。也用作保存图表的文件名。
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数的值
    ))
def benchmark(size, provider):
    """
    对向量加法进行基准测试
    
    参数:
        size: 向量大小
        provider: 'pdl-fp32' 或 'fp32'
    
    返回:
        性能指标 (GB/s)
    """
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    fn = lambda: add(x, y, "pdl" in provider)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles, rep=100)

    # 计算带宽 (GB/s)
    # 3 = 读取 x + 读取 y + 写入 output
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    if supports_pdl():
        print("设备支持程序化依赖启动 (PDL)")
        print("\n=== 正确性验证 ===")
        validate(1024)
        print("\n=== 性能基准测试 ===")
        benchmark.run(print_data=True, show_plots=False)
    else:
        print("此设备不支持 PDL")
        print("PDL 需要计算能力 >= 9.0 的 CUDA 设备(如 Hopper H100 或更新的架构)")

