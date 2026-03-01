# Cursor GPU Skill 介绍

## 背景

最近看到 Cursor 的 skill 机制，其可以给模型提供本地文档上下文，于是就尝试做了一套 GPU 相关的 Skill，把工作中常用的 GPU 相关文档和源码组织起来，提高研发效率。

项目地址: [cursor-gpu-skills](https://github.com/slowlyC/cursor-gpu-skills)

## 快速安装

```bash
git clone https://github.com/slowlyC/cursor-gpu-skills.git
cd cursor-gpu-skills

# 1. 获取外部源码 repo (sparse checkout, ~114MB)
bash update-repos.sh

# 2. 安装 skill 到 ~/.cursor/skills/
bash install.sh
```

安装脚本的逻辑是在 `~/.cursor/skills/` 下创建目录，把 SKILL.md 复制过去（Cursor 不识别软链接的 SKILL.md），其余文件软链接回项目目录。这样修改项目目录里的内容会自动同步，不需要重复安装。

安装完成后 Cursor 会自动识别这些 Skill。在 Prompt 中可以通过 `@cuda-skill`、`@triton-skill` 等方式手动引用。也可以在 SKILL.md 的 description 中写好触发词，在 Agent 模式下自动引用。

## 包含哪些 Skill

目前有四个 Skill，覆盖了 GPU 开发的不同层级:

| Skill | 层级 | 内容 |
|:------|:-----|:-----|
| cuda-skill | 底层 (PTX/CUDA C++) | PTX ISA 9.1、CUDA Runtime/Driver API 13.1、Programming Guide v13.1、Best Practices Guide、Nsight Compute/Systems 文档 |
| triton-skill | 高层 (Python DSL) | Triton/Gluon 教程、生产级内核、语言定义 |
| cutlass-skill | 中间层 (C++/CuTeDSL) | CUTLASS/CuTe 源码、CuTeDSL 示例 |
| sglang-skill | 应用层 (LLM Serving) | SGLang 推理引擎源码、JIT 内核、sgl-kernel |

### cuda-skill

把 NVIDIA 官方文档爬下来转成 Markdown，总共约 800 个文件:

- PTX ISA 9.1 完整规范 (405 files, 2.3MB)，覆盖所有 PTX 指令的语法和语义
- PTX 精简参考 (13 files, 149KB)，来自 Triton 仓库的 `.claude/knowledge`，按功能分类的快速查阅版本
- CUDA Runtime API 13.1 和 Driver API 13.1，函数签名、参数说明、错误码
- CUDA Programming Guide v13.1，从编程模型到 Compute Capabilities 的完整内容
- CUDA C++ Best Practices Guide (73 files, 585KB)，内存优化、执行配置、指令优化等实践指南
- Nsight Compute 完整文档 (9 files, 741KB)，ProfilingGuide、CLI 手册、自定义规则等
- Nsight Systems 完整文档 (5 files, 833KB)，UserGuide、安装指南、分析指南
- 手写的工具指南，包括 nsys/ncu 快速参考、compute-sanitizer、常见性能陷阱

SKILL.md 中定义了搜索策略，AI 会用 Grep 工具在本地文档中检索，而不是把整个文件加载到上下文里。比如查一条 PTX 指令:

```bash
rg "mbarrier.init" ~/.cursor/skills/cuda-skill/references/ptx-docs/9-instruction-set/
```

或者查某个 CUDA API 的用法:

```bash
rg -A 20 "cudaStreamSynchronize" ~/.cursor/skills/cuda-skill/references/cuda-runtime-docs/modules/group__cudart__stream.md
```

文档通过 `scrape_docs.py` 管理，用 `uv run scrape_docs.py all --force` 可以全量更新。

### triton-skill / cutlass-skill / sglang-skill

这三个 Skill 思路相同: 通过 sparse checkout 拉取对应 GitHub 仓库中需要的目录，AI 写代码时可以直接搜索源码找到教程、示例和实现。

以 Triton 为例，拉取的内容包括:
- Triton 教程 (01-11)，从 vector add 到 block-scaled matmul
- Gluon 教程 (01-12)，TMA、wgmma、tcgen05、warp specialization
- 生产级内核: matmul、reduce、top-k、SwiGLU、MXFP
- `triton/language/` 下的语言定义

sparse checkout 控制了只拉必要的目录，避免下载完整仓库。三个 repo 加起来大约 114MB。


## Cursor Skill 机制简介

Cursor 的 Skill 就是放在 `~/.cursor/skills/<skill-name>/SKILL.md` 的 Markdown 文件，文件头有 YAML frontmatter:

```yaml
---
name: cuda-skill
description: "Query NVIDIA PTX ISA 9.1, CUDA Runtime API 13.1..."
---
```

`description` 字段决定了 Cursor 什么时候会自动引用这个 Skill。当用户的问题和 description 匹配时，Cursor 会把 SKILL.md 的内容加入 AI 的上下文。

SKILL.md 里可以写搜索策略、文档路径、使用示例等内容，相当于给 AI 一份"操作手册"。AI 拿到这份手册后，会按照里面的指引用 Grep、Read 等工具去查本地文件，而不是纯靠记忆回答。

有几个需要注意的地方:
- `~/.cursor/skills/<skill-name>/` 目录本身必须是真实目录，不能是软链接
- SKILL.md 必须是真实文件，不能是软链接
- 目录下的其他文件和子目录可以是软链接

## 实际效果

这套 Skill 对以下场景帮助比较明显:

- 查 PTX 指令语法。比如写 inline PTX 时需要查某条指令的操作数格式和约束，以前要在 NVIDIA 文档网站上翻很久，现在 AI 直接在本地搜
- 查 CUDA API 的参数含义和错误码。尤其是 Driver API 那些不太常用的函数
- 写 Triton/Gluon 内核时参考已有的教程和生产级实现
- 查 Compute Capabilities 表，确认某个特性在 sm_90 还是 sm_100 上才支持
- nsys/ncu 的使用方法和指标含义

不足之处也很明显: SKILL.md 能放的信息量有限，复杂的问题仍然需要人工判断和验证。而且这些文档版本是固定的，需要定期更新。

## 总结

如果你也在用 Cursor 做 GPU 开发，可以试试这套 Skill。安装只需要一条命令，不侵入现有环境。
如果有问题或建议可以在 GitHub 上提 Issue。
