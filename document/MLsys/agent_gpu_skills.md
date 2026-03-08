# Agent GPU Skills 介绍

## 背景

Agent Skills 是一种跨工具通用的 AI 上下文扩展机制，通过 `SKILL.md` 文件给模型提供本地文档和源码上下文。Cursor、Claude Code、Codex、Gemini CLI 等主流 AI 编码工具都已支持这一规范。

最近做了一套 GPU 相关的 Skill，把工作中常用的 GPU 文档和源码组织起来，包括 CUDA、Triton、CUTLASS、SGLang 等。原本写 GPU 代码时经常要查 PTX 指令格式、翻 CUDA API 文档、找 Triton 内核写法，这些 Skill 让 AI 可以直接在本地检索这些内容，不用再手动去官网翻文档。

项目地址: [agent-gpu-skills](https://github.com/slowlyC/agent-gpu-skills)

## 快速安装

```bash
git clone https://github.com/slowlyC/agent-gpu-skills.git
cd agent-gpu-skills

# 1. 获取外部源码 repo (sparse checkout, ~114MB)
bash update-repos.sh

# 2. 安装 skill (默认 Cursor，用 --agent claude/codex/gemini 安装到其他工具)
bash install.sh
```

安装脚本在 `~/.cursor/skills/` 下创建真实目录，SKILL.md 复制过去（Cursor 不识别软链接的 SKILL.md），其余文件软链接回项目目录。后续修改项目目录里的内容会自动同步，不需要重复安装。

装好之后，Prompt 中可以通过 `@cuda-skill`、`@triton-skill` 手动引用。也可以不管它，Agent 模式下会根据 SKILL.md 的 description 字段自动触发。

### 其他工具

安装脚本提供了 `--agent` 参数，支持装到其他工具:

```bash
bash install.sh --agent claude     # Claude Code (~/.claude/skills/)
bash install.sh --agent codex      # Codex (~/.agents/skills/)
bash install.sh --agent gemini     # Gemini CLI (~/.gemini/skills/)
```

不过只在 Cursor 下完整验证过，其他工具的 skill 发现和搜索机制可能有差异。

## 包含哪些 Skill

目前四个 Skill，从底层到应用层:

| Skill | 层级 | 内容 |
|:------|:-----|:-----|
| cuda-skill | 底层 (PTX/CUDA C++) | PTX ISA 9.1、CUDA Runtime/Driver API 13.1、Programming Guide v13.1、Best Practices Guide、Nsight Compute/Systems 文档 |
| cutlass-skill | 中间层 (CUTLASS/CuTeDSL) | CUTLASS/CuTe 源码、CuTeDSL 示例 |
| triton-skill | 高层 (Python DSL) | Triton/Gluon 教程、生产级内核、语言定义 |
| sglang-skill | 应用层 (LLM Serving) | SGLang 推理引擎源码、JIT 内核、sgl-kernel |

### cuda-skill

NVIDIA 官方文档爬下来转成 Markdown，总共约 800 个文件:

- PTX ISA 9.1 完整规范 (405 files, 2.3MB)，覆盖所有 PTX 指令的语法和语义
- PTX 精简参考 (13 files, 149KB)，来自 Triton 仓库的 `.claude/knowledge`，按功能分类的快速查阅版
- CUDA Runtime API 13.1 和 Driver API 13.1，函数签名、参数说明、错误码
- CUDA Programming Guide v13.1，从编程模型到 Compute Capabilities 的完整内容
- CUDA C++ Best Practices Guide (73 files, 585KB)，内存优化、执行配置、指令优化
- Nsight Compute 文档 (9 files, 741KB)，ProfilingGuide、CLI 手册、自定义规则
- Nsight Systems 文档 (5 files, 833KB)，UserGuide、安装指南、分析指南
- 手写的工具指南，nsys/ncu 快速参考、compute-sanitizer、常见性能陷阱

AI 不会把这些文件全塞进上下文，而是按 SKILL.md 里定义的搜索策略，用 Grep 在本地文档中检索。比如查一条 PTX 指令:

```bash
rg "mbarrier.init" ~/.cursor/skills/cuda-skill/references/ptx-docs/9-instruction-set/
```

或者查某个 CUDA API 的用法:

```bash
rg -A 20 "cudaStreamSynchronize" ~/.cursor/skills/cuda-skill/references/cuda-runtime-docs/modules/group__cudart__stream.md
```

文档用 `scrape_docs.py` 管理，`uv run scrape_docs.py all --force` 可以全量更新。

### cutlass-skill / triton-skill / sglang-skill

这三个 Skill 思路相同: 通过 sparse checkout 拉取对应 GitHub 仓库中需要的目录，AI 写代码时直接搜索源码找教程和实现。

以 Triton 为例，拉取的内容包括:
- Triton 教程 (01-11)，从 vector add 到 block-scaled matmul
- Gluon 教程 (01-12)，TMA、wgmma、tcgen05、warp specialization
- 生产级内核: matmul、reduce、top-k、SwiGLU、MXFP
- `triton/language/` 下的语言定义

sparse checkout 只拉必要的目录，不下载完整仓库。三个 repo 加起来约 114MB。

## Agent Skill 机制简介

Agent Skill 就是放在工具 skill 目录下的 `SKILL.md` 文件，带 YAML frontmatter:

```yaml
---
name: cuda-skill
description: "Query NVIDIA PTX ISA 9.1, CUDA Runtime API 13.1..."
---
```

`description` 字段决定了什么时候自动触发。用户的问题和 description 匹配时，工具会把 SKILL.md 的内容加入上下文。

SKILL.md 里写搜索策略、文档路径、使用示例——AI 拿到后按指引用 Grep、Read 等工具查本地文件，而不是纯靠记忆回答。

这个规范 Cursor、Claude Code、Codex、Gemini CLI、GitHub Copilot、Windsurf 都支持，路径不同但格式一致:

| 工具 | 项目级路径 | 全局路径 |
|:-----|:-----------|:---------|
| Cursor | `.cursor/skills/` | `~/.cursor/skills/` |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| Codex | `.agents/skills/` | `~/.agents/skills/` |
| Gemini CLI | `.gemini/skills/` | `~/.gemini/skills/` |

Cursor 下有几个安装注意点:
- skill 目录本身必须是真实目录，不能是软链接
- SKILL.md 必须是真实文件，不能是软链接
- 目录下的其他文件和子目录可以是软链接

## 实际效果

日常用下来，几个场景比较受益:

- 写 inline PTX 时查指令的操作数格式和约束，以前在 NVIDIA 文档网站上翻很久，现在 AI 直接在本地搜到
- 查 CUDA API 的参数含义和错误码，尤其 Driver API 那些不太常用的函数
- 写 Triton/Gluon 内核时参考已有的教程和生产级实现
- 确认某个特性在 sm_90 还是 sm_100 上才支持
- nsys/ncu 的命令行参数和指标含义

当然 SKILL.md 能放的信息量有限，复杂问题仍然需要人工判断。文档版本也是固定的，需要定期更新。

## 总结

做 GPU 开发可以试试这套 Skill，安装一条命令搞定，不侵入现有环境。目前在 Cursor 下验证过，SKILL.md 格式也兼容 Claude Code、Codex 等工具。

有问题或建议欢迎留言，或在 [GitHub](https://github.com/slowlyC/agent-gpu-skills) 上提 Issue。
