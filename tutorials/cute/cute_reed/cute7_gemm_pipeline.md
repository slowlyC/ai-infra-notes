# cute 之 GEMM流水线

**Author:** [reed](https://www.zhihu.com/people/reed)

**Link:** [https://zhuanlan.zhihu.com/p/665082713](https://zhuanlan.zhihu.com/p/665082713)

---

前面文章我们介绍了CuTe的[Copy抽象](https://zhuanlan.zhihu.com/p/666232173)、[MMA抽象](https://zhuanlan.zhihu.com/p/663092747)，基于这些抽象进行了[简单的GEMM实现](https://zhuanlan.zhihu.com/p/667521327)。从逻辑上而言CuTe的介绍已经结束了，但对于GEMM运算还有一项重要的优化需要考虑：如何高效地、并行地利用GPU中的数据加载和计算单元，即如何组织Copy和MMA抽象以完成高效的GEMM计算。这部分内容属于GEMM的策略层面，不是CuTe的功能范畴。为了系列文章标题的对称性，我们仍将题目取为"CuTe之GEMM流水线"，但需要明确的是，流水线本身不属于CuTe，而是GEMM的优化策略。

文章结构：首先通过回顾经典的RISC硬件指令流水线引入流水线对性能提升的作用; 然后通过类比介绍GEMM算法常用的软件流水线（Tile间和Tile内）; 接着介绍NVIDIA Ampere架构提供的异步拷贝指令和MultiStage流水线; 最后总结GEMM流水线与CuTe的关系。

## RISC硬件流水线

在当今的处理器微架构中，流水线技术是提升指令并行的核心技术。流水线处理器是指将每一个指令的执行过程分为多个阶段（Stage），并且允许不同指令的不同阶段可以被同时处理。以经典的RISC（Reduced Instruction Set Computer）流水线为例，一条指令的执行被分为五个阶段：

* 取指（IF = Instruction Fetch），从指令缓存中根据程序运行的位置（PC = program counter）取出一条要执行的指令; 
* 译码（ID = Instruction Decode），将取出的二进制编码分解成要进行的运算类型、源寄存器和目的寄存器; 
* 执行（EX = EXecute），执行单元执行特定的运算; 
* 访存（MEM = MEMory），如果指令有对内存的访问需求，该阶段则负责相应的内存读写; 
* 写回（WB = Write Back），将执行单元的执行结果和（或）内存的访问结果写出到目的寄存器。

![](images/cute7_gemm_pipeline_001.jpg)
*Figure 1. 流水线和非流水线处理器指令执行的时间分析*

图1对比了非流水线和流水线架构执行指令时的时间对比，可以看到非流水线结构执行每一条指令都需要执行所有的阶段，其执行三条指令（Inst-1, Inst-2, Inst-3）所需要的时间如图上半部分所示。在流水线结构下，第一条指令在做取指后进入译码阶段，这时候第二条指令则可以进入取指阶段，后续的指令阶段也是类似的可以产生重叠。如图中的下半部分所示，以流水线的形式执行三条指令所需的时间要比非流水线的模式小很多。流水线模式提升了指令执行中的各个阶段的不同单元的使用率，使得每一个时刻每一个单元都能充分利用，而不是非流水线结构中一个时间点只有一部分单元运行，而其他时间都在空闲等待。

## GEMM软件流水线（Tile间）

指令流水线通过硬件设计提升了各个单元的利用率，提升并行度继而提升运行效率。对于GEMM问题而言，我们也可以采用这种思路，通过软件编程来实现更好的并行。

![](images/cute7_gemm_pipeline_002.jpg)
*Figure 2. 循环k模式的矩阵乘法的指令构成*

如图2所示，一个典型的sliced-k模式的GEMM实现中，通过循环K轴方向的tile来累加得到最终的CTile的结果。则类似RISC中的流水线模式，我们将计算每一个Tile的矩阵乘积作为一个基本的单元（如RISC中的指令），则该指令的执行类比RISC流水线可以划分为多个阶段，

* 数据加载到共享内存（LDGSTS = LoaD Global STore Shared memory）
* 数据加载到寄存器（LDSM = LoaD Shared Matrix）
* 块状矩阵乘法运算（MMA = Matrix Multiply Accumulate）

其中第一个阶段的输出数据存放在共享内存中，第二个阶段的数据存放在寄存器中。和RISC中的流水线类似，我们把一个Tile的计算过程分成三个阶段，如果各个阶段可以重叠则效率可以极大地提升，于是有了流水线思路优化的GEMM执行效果，如图3所示，

![](images/cute7_gemm_pipeline_003.jpg)
*Figure 3. 非流水线模式和流水线模式下的GEMM执行逻辑*

这样三个阶段的执行便可以并行起来，通过流水线使得各个单元能够同时工作，提高了各个单元的利用效率，包括全局内存到共享内存的数据加载、共享内存到寄存器的数据加载、矩阵计算，从而提升GEMM的运行效率。

## Tile内的流水线

![](images/cute7_gemm_pipeline_004.jpg)
*Figure 4. GEMM Tile内的流水线模式*

矩阵乘法中的分块模式在Tile内同样可以使用流水线（本文称为Tile内小k循环）。如图4所示，对于Tile级别的矩阵乘，一般一个Tile内包含的矩阵大小需要若干条指令（MMA_Atom中的指令）才能完成计算，并且各个矩阵乘法的输入数据相互独立。因此可以将数据加载和计算组成流水线，提高数据加载单元和计算单元的利用率。如图中pipelined标记的部分所示，这形成了Tile内的流水线模式（二级流水线），通过重叠数据加载和计算提高Tile内矩阵计算的整体效率。

## 异步拷贝和MultiStage流水线

为了提升数据加载效率，NVIDIA在Ampere架构的GPU中提供了异步拷贝指令`cp.async`（SASS汇编为LDGSTS = LoaD Global Store Shared）。该异步拷贝指令可以异步地完成全局内存到共享内存的数据加载。在Ampere架构之前，全局内存到共享内存的数据加载必须经过寄存器，在寄存器层面会产生数据依赖。由于GPU的顺序发射机制和scoreboard依赖跟踪（in-order issue, in-order execute），全局内存到共享内存的数据传输因寄存器依赖而引入stall。Ampere提供的`cp.async`克服了这个约束，直接实现全局内存到共享内存的加载，无需经过寄存器。由于数据是异步加载的，即指令发射后便可执行后续指令而无需等待，该架构提供了commit和wait机制来做显式同步：commit用于标记事务的同步点，wait用于同步到特定同步点，保证某个同步点之前的数据都已拷贝完成。

![](images/cute7_gemm_pipeline_005.jpg)
*Figure 5. 异步拷贝机制*

如图5所示，我们通过`cp.async`指令提交了三个全局内存到共享内存的拷贝任务，同时通过commit提交了三个事务点和两个wait，其中`wait<1>`表示可以允许最多有一个未完成的异步事务（G2 -> S2），即`wait<1>`执行结束能够确保G1到S1的拷贝已经完成，`wait<0>`表示允许有零个未完成的事务。也就是其会等待之前所有的commit的任务都完成，即G1->S1, G2->S2, G3->S3全部已经完成。

有了异步数据拷贝指令我们便可以完成全局内存到共享内存的异步加载，即完成矩阵A、B Tile的加载，再整合Tile间和Tile内的流水线，我们可以得到GEMM计算的MultiStage流水线模型，如图6所示，

![](images/cute7_gemm_pipeline_006.jpg)
*Figure 6. MultiStage流水线模型*

图中标注了三类操作：

* 浅绿色 $G^i \rightarrow S^i$：全局内存到共享内存的异步数据加载，大小为一个Tile（合并表示TileA和TileB的加载，对应Tile循环）
* 棕色 $S_j \rightarrow R_j$：共享内存到寄存器的数据加载，对应Tile内小矩阵的加载（即Tile内小k循环）
* 深绿色 $\text{mma}(R_i)$：寄存器上的矩阵乘法计算（同属Tile内小k循环）

mma区域的两条黑色边界线表示Tile内小k循环的起点和终点，黑线之间完成Tile内的矩阵乘法; 连接两条边界线的虚线曲线表示完成一个Tile的计算后继续下一个Tile。以图示kStage = 4为例，MultiStage流水线的执行过程如下：

**Prologue阶段**（首个Tile计算前）：发射 stage - 1 = 3 个异步的全局内存到共享内存加载任务（G0→S0, G1→S1, G2→S2）。所有异步任务发射后，执行wait等待S0完成，确保第一个Tile的数据已到达共享内存。随后从S0中取出 ik = 0 的矩阵计算所需数据到寄存器R0（对应图中第一条黑色虚线和第一条黑色实线之间的部分）。之后进入**Tile内小k循环**，此时R0已就绪，可以发射异步加载新Tile数据（如G3→S3），进入循环后执行二个动作：

1. 从共享内存读取下一个小k矩阵乘法所需数据到寄存器（如R1）
2. 执行当前小k的矩阵运算 mma(R0)

其中共享内存读出的数据与mma所需数据的依赖关系通过图中箭头表示，以流水线方式完成数据加载和计算。

**Tile边界处理**：在最后一个小k循环之前，需要读取下一个Tile中第一个小k的数据（共享内存到寄存器）。但此时下一个Tile的数据（全局内存到共享内存）尚未确认完成，需要插入对S1的异步事务等待（wait S1）。等待结束后，从S1中加载下一个Tile的第一个小k数据到寄存器R0（注意此处的共享内存已是下一个Tile，即S1），然后完成当前Tile最后一个小k的mma计算。至此Tile内小k循环结束，继续下一个Tile的计算，最终完成整个Tile循环。

如上便是MultiStage的GEMM流水线（Tile间多级，Tile内二级），其中multi表示多个，具体的个数即shared memory中间buffer的个数。例如stage为5的GEMM流水线表示有五个shared memory buffer，每个buffer可以存放一个Tile的数据（包含TileA和TileB）。Tile循环开始前先发射 stage - 1 个全局内存到共享内存的加载，然后在循环中加载下一个Tile，循环使用这些buffer完成所有数据的加载。在没有异步拷贝支持的GPU架构中，寄存器的依赖关系和`__syncthreads`的全局同步影响，决定了最多只能有两个memory buffer（实质是寄存器buffer），一个用于当前数据的计算，一个用于后续数据的加载，这就是常说的双缓冲机制（double buffer），可以认为是MultiStage中 stage = 2 的特例。

合适的stage数量本质是数据加载能力和矩阵计算能力的平衡，由Tile大小和硬件latency共同决定。具体选择时可以通过micro-benchmark获取指令latency来正向设计，也可以在具体环境中试验tuning得到。后面的文章中，我们会利用这种软件流水线方式实现更高效的GEMM。

## 总结

回顾上面的流水线建立过程不难发现，软件流水线的本质是：合理地控制数据搬运和计算的粒度与执行顺序，使得背后的硬件单元能够更充分地并行执行。CuTe提供了数据搬运和计算两方面的抽象（Copy和MMA），按照流水线的思路组织这些Copy和MMA即可完成高效的矩阵乘法。也就是说CuTe是工具，如何更好地使用这些工具来达到更好的硬件利用则属于设计，已经超越了CuTe的范畴。后续文章我们会使用CuTe提供的Copy和MMA抽象，基于本文讨论的流水线模式实现高效的GEMM。

## 参考

1. [Classic RISC pipeline - Wikipedia](https://en.wikipedia.org/wiki/Classic_RISC_pipeline)
2. [Computer Organization and Design (RISC-V Edition) - Springer](https://link.springer.com/book/10.1007/978-3-031-01729-2)
3. [CUTLASS sm80_mma_multistage.hpp - GitHub](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm80_mma_multistage.hpp)
