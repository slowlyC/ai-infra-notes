# NVidia GPU指令集架构-Load和Cache

**Author:** [reed](https://www.zhihu.com/people/reed)

**Link:** [https://zhuanlan.zhihu.com/p/692445145](https://zhuanlan.zhihu.com/p/692445145)

---

前文介绍了 [NVidia GPU 指令集架构](https://zhuanlan.zhihu.com/p/686198447)中的[寄存器部分](https://zhuanlan.zhihu.com/p/688616037)。对于 GPU 程序而言，寄存器中的数据最初来自外部存储结构，数据如何从外部存储搬运到寄存器、搬运过程中经过哪些 Cache，对程序效率有重要影响。本文围绕数据搬运展开，重点介绍 NVidia GPU 的存储层次与 Cache 层级，以及各层级之间的数据搬运指令和 Cache 控制指令。文章首先介绍计算单元与存储层级的组织关系，然后介绍 Cache 和 Shared Memory 机构，接着介绍 Load 和 Cache 相关指令及预取行为，最后进行总结。

## NVidia GPU的计算单元和存储层级

GPU 是一个高度并行的设备，装配了大量计算单元。为了更好地组织这些计算单元、充分利用数据局部性（Spatial Locality）并支持数据规约与同步，GPU 采用层次化的组织形式。在 Ampere 及之前的架构中，计算单元的组织分为三个层次：最小的计算机构是 SubCore，负责线程束（warp）的执行；4 个 SubCore 组成一个 SM（Stream Multiprocessor）；多个 SM 组成 GPU 设备（device）。不同设计规格的设备，SM 数目是最直观的区别，如数据中心级 A100 和消费级 GeForce RTX 3080/3090 都是 Ampere 架构，计算单元层面最大的区别就是 SM 数目。

将 SM 划分为多个 SubCore 而非做成单一大核，有几方面的考量。每个 SubCore 拥有独立的 warp scheduler，可以从自己管理的 warp 池中选择 ready 的 warp 发射指令，4 个 SubCore 独立调度降低了单个调度器的复杂度和功耗。寄存器文件按 SubCore 物理分区，减少了端口争用。在延迟隐藏方面，每个 SubCore 分配到多个 warp（如 Ampere 上每个 SubCore 最多管理 16 个 warp），当一个 warp 因访存而 stall 时，scheduler 可以零开销切换到另一个 ready 的 warp（寄存器状态始终驻留，不需要 save/restore），这正是 GPU 延迟隐藏的基本机制。

就 SubCore 内部而言（如图 1），核心计算单元包括 Tensor Core（矩阵计算）、CUDA Core（向量化浮点和整数乘加）、SFU（Special Function Unit，完成 sin、exp2、sqrt、rcp 等超越函数），此外还有寄存器文件（Register File）、具有广播语义的常量内存（图中标记为 constant cache）以及外部数据加载存储用的 Load Store Unit。SubCore 中还有 warp scheduler、branch unit、FP64 等单元，由于它们对理解指令集架构的作用不太直接，图中未单独标出。

存储层次方面，SubCore 内有寄存器和 Constant Cache；SM 内有 4 个 SubCore 共享的 L1 Cache 和 Shared Memory；所有 SM 通过交叉开关（CrossBar）共享 L2 Cache；分 Slice 的 L2 Cache 通过 Memory Controller 连接外部存储（独立 Die 上的 HBM 或 GDDR）。数据中心级 A100 和消费级 GeForce RTX 3080 的另一个重要区别是内存介质技术（HBM vs GDDR）。

![](images/03_NVidia_GPU指令集架构-Load和Cache_001.jpg)
*Figure-1. NVidia GPU存储层级和Cache层级*

如图 2 所示，软件是对物理硬件的抽象。对于 CUDA 编程而言，物理的 SubCore 大致对应 warp，SM 对应 thread block，多个 SM 组成的 device 对应 grid。软件到硬件的映射中多了调度逻辑：一个物理 SubCore 可以运行多个 warp，一个 SM 可以运行多个 thread block，有限的 SM 组成的 device 可以运行远超硬件数目的 grid。

![](images/03_NVidia_GPU指令集架构-Load和Cache_002.jpg)
*Figure-2. Hardware and its Software Abstraction*

## GPU中的Cache和Shared Memory机构

Cache 是利用时间局部性和空间局部性提升访存效率的经典方案。在 CUDA 编程中，数据最初存储在全局内存（global memory）中，而核心计算发生在 SubCore 的计算单元内。前文提到 GPU 是 Load-Store 架构（寄存器-寄存器架构），计算单元只能访问 SubCore 内的寄存器（和 constant cache），需要外部数据时必须通过 Load 指令将数据加载到寄存器。GPU 利用局部性原理，在 global memory 和寄存器之间设置了两层 Cache：L2 Cache 和 L1 Cache。L2 Cache（A100 上为 40MB）被所有 SM 共享；L1 Cache 装配在每个 SM 中，被 SM 内 4 个 SubCore 共享。

当 SubCore 的 Load Store Unit 发出全局内存访问请求时，先查 L1 Cache，命中则直接返回；L1 miss 则请求 L2 Cache，L2 命中则返回；只有两级 Cache 都 miss 时才访问全局内存。L1/L2 Cache 的带宽和访问延迟远优于 HBM，合理调整数据访问模式、提升局部性可以更充分地利用各级 Cache，显著提升程序效率。

Cache 是不可编程的存储空间，命中和替换逻辑由硬件自行控制。有时程序员可以更精准地管理共享数据，选择更合适的时机进行更新和同步。为此 NVidia GPU 在 SM 级别提供了可编程的 Shared Memory，它是一片可寻址的地址空间，支持 Load/Store 以及数据可见性同步。需要反复使用的数据可以显式加载到 Shared Memory 中，避免重复访问低层级内存。从 Fermi 架构开始，L1 Cache 和 Shared Memory 共享同一份后端 SRAM 存储，仅在前端做 tag 命中和地址判定，用户可以根据场景灵活配置两者的容量分配。

结合上面的介绍我们不难发现，从不同的维度去看数据加载有不同的形式：

1. 软件概念：Global -> Shared -> Register
2. 物理概念：HBM(Global) -> SRAM(L2 Cache) -> SRAM(L1 Cache) -> SRAM(Register File)
3. 片上概念：OffChip(Global) -> OnChip(L2 Cache) -> OnChip(L1 Cache) -> OnChip(Register File)
4. 共享层次：SMs(Global) -> SMs(L2 Cache) -> SM(L1 Cache) -> SM's SubCore(Register File)

## 数据Load指令

数据加载相关的指令整体如下：

```text
LD, LDG, LDS, LDSM, LDL
```

其中 LD（LoaD）是通用加载指令，用于编译器无法在编译时推导地址空间类型的情况。如果编译时能确定地址空间，则使用带类型的加载指令：LDG（LoaD Global memory）、LDS（LoaD Shared memory）、LDSM（LoaD Shared Matrix）、LDL（LoaD Local memory）。

**全局内存到寄存器：**

```text
LDG.类型.向量.Cache控制.L2预取
```

在介绍各条指令之前，先说明一个贯穿 SASS 指令集的概念：**Modifier**。Modifier 是指令助记符中以 `.` 分隔的后缀标记，每个 . 后面的部分都是一个 modifier，用于指定指令行为的各种细节。例如 `LDG.E.128.EF` 中，`.E` 表示 64 位扩展地址，`.128` 表示加载 128bit 数据，`.EF` 控制 Cache 策略（evict-first）。不同类别的指令有不同的 Modifier 集合，编译器根据源码语义选择合适的组合。

LDG 指令将全局内存中的数据加载到寄存器。通过 Modifier 可以配置加载的数据宽度（8bit、16bit、128bit 等）、各层级 Cache 的 bypass 策略以及是否对 L2 进行预取。向量化加载（大字长加载）如 LDG.128 是 NVidia GPU 支持的最宽加载指令，一条指令加载 128bit 数据。对同等规模的数据，使用更宽的加载指令可以减少 warp 的指令调度次数、降低调度开销、减少 MIO queue 的事务数，从而避免 queue 满造成的阻塞。除了单指令位宽，更高效的数据加载还需要考虑合并访存（关于合并访存的优势，后续会在"软件优化的硬件解释"系列中详细介绍）。

**全局内存到共享内存：**

```text
LDGSTS, LDGDEPBAR, DEPBAR.LE SB0, 0x1
```

LDGSTS（LoaD Global memory STore Shared memory）实现从全局内存到共享内存的异步数据搬运，数据不经过寄存器，可以减少寄存器占用和依赖。这条指令在矩阵计算（尤其是 Multi Stage GEMM 流水线）中有重要作用，可参考 [cute 之 GEMM 流水线](https://zhuanlan.zhihu.com/p/665082713)的异步拷贝章节。该指令需要配合 Barrier 设置和等待指令（LDGDEPBAR、DEPBAR）协同使用，加载时可以指定是否在 L1 进行 Cache 以及对 L2 进行预取。

**共享内存到寄存器：**

```text
LDS.类型.向量化
LDSM.块.转置
```

LDS 的 modifier 可以设置数据位宽，与 LDG 类似，高位宽指令可以减少 warp 的指令调度数和 MIO queue 中的事务数目，避免 queue 满引起的阻塞。LDSM 是 warp 级协作指令，warp 内 32 个线程协作地从共享内存加载数据到寄存器，加载模式专门适配 Tensor Core 指令（HMMA/IMMA）的操作数布局，比用 LDS 加载再手动 shuffle 重排高效得多。CUDA PTX 中的 `ldmatrix` 指令编译到 SASS 后对应的就是 LDSM，例如 `ldmatrix.sync.aligned.x4.m8n8.shared.b16` 编译后变为 `LDSM.16.M88.4`，其中 modifier `.16` 表示数据宽度，`.M88` 对应 m8n8 布局，`.4` 对应 x4（加载 4 个 fragment）。更细节的介绍可参考 [cute 之 Copy 抽象](https://zhuanlan.zhihu.com/p/666232173)和 [ldmatrix 指令优势介绍](https://www.zhihu.com/question/600927104/answer/3029266372)。

**局部数组和寄存器溢出：**

```text
LDL
```

有三种情况会引入 Local Memory：1. 线程使用局部数组且数组下标不能在编译时确定；2. 单线程寄存器使用数超过 255；3. 访问 kernel 中的数组常量时使用了编译时无法确定的索引。Local Memory 是 CUDA 编程中的概念，它的物理实体是全局内存中的一段。当上面情况发生时，每一个线程都会被分配一段全局内存来作为数据空间，由于数据需要对全局内存进行读写，在线程数较多的场景下开销很大，应尽量避免使用。

## 广播语义的常量Cache

除了 Load 全局内存和共享内存，SubCore 中还有 constant cache 机构。虽然名字叫 cache，但它本质上更接近寄存器：提供广播语义，当 warp 内所有线程访问同一个 constant 位置时，访问速度与寄存器相当，因此可以直接编码在指令操作数中（如 `c[0x0][0x180]`）。kernel 参数需要广播给所有执行线程，正是通过 constant cache 实现的。用户也可以声明 device 端的可编程常量（`__constant__ __device__ int a;`）。

当 warp 内不同线程访问不同的 constant 位置时，访问会串行化，延迟不再固定。为此 SASS 提供了 LDC 指令，用于处理不同线程访问不同 constant 位置的场景。

## 寄存器reuse和Prefetch

除了以上常规的存储机构和 Cache，计算单元流水线中已经加载的数据也可以复用，体现为寄存器的 reuse。可以把它理解为寄存器 cache，能减少寄存器文件的读端口压力并在一定程度上降低功耗，如：

```text
R1.reuse
```

除了 Load 指令可以做伴随的数据预取，SASS 还提供了显式的 Cache 预取指令（CCTL = Cache ConTroL），如：

```text
CCTL.E.PF2
```

## 总结

本文介绍了 NVidia GPU 的内存层次、Cache 机构以及指令集架构中 Load 和 Cache 相关的指令。了解这些指令有助于理解硬件在数据搬运时的行为，充分合理地利用各级 Cache 和搬运指令可以提升数据搬运效率，指导性能优化。

## 参考

[https://www.qidian.com/book/1031795831/](https://www.qidian.com/book/1031795831/)

[超标量处理器设计: 9787302347071: 姚永斌: Books](https://www.amazon.com/超标量处理器设计-姚永斌/dp/B00JFJTI2I/ref=monarch\_sidesheet)

[https://pc.watch.impress.co.jp/docs/column/kaigai/1275220.html](https://pc.watch.impress.co.jp/docs/column/kaigai/1275220.html)

[https://pc.watch.impress.co.jp/video/pcw/docs/1275/220/p3.pdf](https://pc.watch.impress.co.jp/video/pcw/docs/1275/220/p3.pdf)

[reed：cute 之 GEMM流水线](https://zhuanlan.zhihu.com/p/665082713)

[reed：cute 之 Copy抽象](https://zhuanlan.zhihu.com/p/666232173)

[tensorcore中ldmatrix指令的优势是什么？](https://www.zhihu.com/question/600927104/answer/3029266372)
