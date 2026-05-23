# Hermes 学习记录

## 2026-05-23 23:30 - v0 RAG 系统骨架搭建

### 做对的
- brainstorming 阶段调研了 RAGFlow/Dify/FastGPT/AnythingLLM 四个成熟项目，借鉴了 provider 按能力抽象 + pipeline 化 ingest 的设计
- 用户 review 发现 5 个 must-fix（目录结构冲突、ChromaDB metadata 约束、document_id 冗余、lifespan 定义、日志缺失）全部在实现前修复
- TDD 执行顺利，33 个测试全部通过，每个 task 都有红→绿→commit 循环

### 做错的 / 被用户纠正的
- 初始架构用 `app/llm/` 目录，与项目 CLAUDE.md 冲突（违反 Karpathy #3 手术刀原则：没先看清现有规约）
- 设计文档 Chunk.metadata 定义与 VectorStoreBase.add() 注释矛盾（A/B 两处不一致，用户二次 review 才发现）
- loguru kwargs 问题：我确认 loguru 支持 kwargs 是对的，但用户指出 extra 字段不会进入默认格式化输出（实际是 bind() 才能结构化）
- plan 里 BGE 测试 mock 返回 list 而非 numpy array，MIMOLLM 测试用 MagicMock 而非 AsyncMock — 都是用户 review 时发现的，不是我自己 catch 的

### 教训提炼
- 设计文档写完后做 self-review 时，不要只查"有没有写"，要查"写的两处是否一致" — 特别是先写了 A 部分后来改了 B 部分的场景
- plan 里 mock 的类型要与实现层的返回类型严格匹配（numpy array vs list, MagicMock vs AsyncMock）— 这种 bug 只有在真正跑测试时才会爆，self-review 看不出来
