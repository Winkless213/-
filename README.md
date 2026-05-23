# 本地企业知识库系统(原型)

## 项目目标

搭建一个本地运行的企业知识库系统(RAG 架构),学习与原型用途。
后续可喂入真实企业文档(SINOBA 的采购数据、客户档案、技术规格等)。

**v0 重点是搭建系统骨架,不依赖真实数据**。

## 核心场景

- 用户上传 markdown / 文本文档 → 系统自动 ingest(切片 + 向量化 + 入库)
- 用户提问 → 系统检索相关片段 → 调用 LLM 生成答案
- 系统记录每次查询的来源 chunks(可解释性)

## 技术栈(初步选型,/clarify 阶段可讨论调整)

- 后端:Python 3.11+ / FastAPI
- 向量库:ChromaDB(本地、零配置)
- LLM:MIMO(API)
- Embedding:优先 MIMO,如不支持则 BGE-small-zh-v1.5(本地跑)
- 测试:pytest
- 部署:本地 uvicorn,不做 Docker

## 范围与约束

### v0 范围
- 系统骨架完整、可运行
- 自带 3-5 篇合成 markdown 测试文档,启动时自动 ingest 验证链路
- 两个核心端点:`/ingest`、`/query`
- 单用户本地、无鉴权
- 不做前端,用 FastAPI 自带 `/docs` 测试

### v0 明确不做
- 不接真实数据
- 不做权限 / 多用户
- 不做前端 UI
- 不支持 PDF / Word / Excel
- 不做对话历史 / 多轮上下文
- 不做监控、日志聚合
- 不做 Docker 化

## 后续路线

| 版本 | 增量 |
|---|---|
| v0 | 系统骨架 + 合成数据验证(当前) |
| v1 | 喂入第一批真实数据 |
| v2 | 支持 PDF / Word |
| v3 | 简单前端 UI |
| v4 | 接入 Xunhuoxia(作为 MCP 工具) |

## 工程原则

- 所有 LLM / Embedding 调用封装在 `app/llm/`
- 所有 prompt 集中放 `app/prompts/`
- 配置统一从 `.env` 读取
- 关键路径必须有单元测试
- README 必须提供"如何喂入你自己的数据"章节

## 工作流(给 Claude Code)

1. 阅读本 README,理解项目目标与范围
2. 阅读 `CLAUDE.md` 了解工程规约
3. 等待用户 `/clarify` 命令开始需求澄清
4. **未经用户明确 "go" 不要开始实现**
