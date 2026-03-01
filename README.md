# ai-infra-notes

> 施工中

GPU 编程与 AI 基础设施笔记。内容覆盖 GPU kernel 编写（Triton、Gluon、CuTe/CuTeDSL）、GPU 硬件架构、分布式训练，以及 MLsys 领域的论文/代码分析。

## 目录结构

```
ai-infra-notes/
├── tutorials/          # 代码教程（带详细中文注释的 .py / .cu 源码）
│   ├── triton/         # Triton GPU kernel 教程 (11 篇)
│   ├── gluon/          # Gluon DSL 教程 (12 篇, 面向 Blackwell)
│   ├── cutile/         # Cutile kernel 教程 (8 篇)
│   ├── CuTeDSL/        # CuTeDSL 教程 (Ampere/Hopper/Blackwell + Notebooks)
│   └── cute/           # CuTe C++/CUDA 教程
└── document/           # 文档与分析文章
    ├── dsl/            # Triton / Gluon 教程的 Markdown 文档版
    ├── MLsys/              # MLsys 论文与代码分析
    └── distributed_training/  # 分布式训练技术详解
```

## Tutorials

- **Triton** — 从 Vector Addition 到 Persistent Matmul，覆盖 Triton kernel 编写的完整学习路径
- **Gluon** — 面向 Blackwell GPU 的 Gluon DSL 教程，从 Layout、TMA、WGMMA 到完整的 Flash Attention kernel
- **Cutile** — Cutile 框架的常见算子实现
- **CuTeDSL** — 按架构分类的 CuTeDSL kernel 教程（Ampere / Hopper / Blackwell / Distributed），附带 12 篇中文 Jupyter Notebook
- **CuTe C++** — CuTe 的 C++/CUDA 原生教程

## Document

- **DSL 教程文档** — 与 tutorials 中的 Python 源码一一对应，将代码注释整理为可阅读的 Markdown 文档
- **MLsys** — GPU架构、FlashAttention 3 深度解析、GDN 分析、Gluon Prefill 优化等
- **Distributed Training** — 序列并行、流水线并行详解
