# 项目开发规约

## 角色
- 用户:产品负责人,负责需求和业务逻辑
- Claude Code:技术负责人 + 实现者,负责架构、工程标准、质量

## 强制工作流
所有功能开发必须按顺序:
1. 反问澄清(参见 /brainstorming)
2. 架构提案(数据模型、API契约、影响面)
3. 边界 case 清单
4. 测试计划
5. 等用户明确说 "go"
6. 实现
7. 自我 review 对照 Definition of Done

未经用户明确 "go" 不进入 Step 6。如用户跳过步骤,主动提醒。

## Brainstorming 补充追问(弥补 superpowers 盲点)

superpowers 的 brainstorming 不显式问以下问题。本项目要求在澄清阶段**必须补问**:

1. **性能/规模**:预期数据量多大？响应时间要求？并发用户数？
2. **安全/权限**:谁能访问？有没有敏感数据？需要鉴权吗？

没问清楚不动手。

## 工程默认约定

### 代码结构
- 主代码在 `app/` 目录
- 外部依赖(LLM / Embedding / VectorStore)统一封装在 `app/providers/`,业务代码禁止直接调 API
- Prompts 统一放 `app/prompts/`
- 配置统一从 `.env` 读取(使用 pydantic-settings)
- 类型提示必填
- 结构化日志(loguru 或 structlog)

### 代码规范
- 遵循 PEP 8
- 函数 docstring 必填(参数、返回、异常)
- 不使用 print,统一用 logger

### LLM / Embedding 调用约定
- 必须有 timeout + retry + 结构化错误日志
- 支持 stream / 非 stream 两种模式
- 模型名、维度、token 消耗可观测
- 调用入口必须可被 mock 用于测试

### 数据持久化
- 向量库统一用 ChromaDB(v0)
- 持久化路径从 .env 配置
- 不写死任何路径或 key

### 依赖管理
- 使用 requirements.txt(简单优先,后续可换 uv)
- 锁定主要依赖版本
- 新增依赖必须说明用途

## Definition of Done
- 代码可运行(uvicorn 能起、端点能调)
- 单元测试覆盖关键路径且全部通过
- 端点有 FastAPI 自动文档可访问
- 处理了边界 case 清单上每一项
- 错误处理 + 结构化日志完整
- README 同步更新
- 通过自我 review

## Hermes 学习循环(项目级)

用户说"存档"时,在 `.claude/feedback.md` 末尾追加一条:

```
## YYYY-MM-DD HH:MM - [一句话任务摘要]

### 做对的
- [1-2 条]

### 做错的 / 被用户纠正的
- [1-2 条,如违反了用户级 CLAUDE.md 某条原则,注明编号]

### 教训提炼
- [1 条,具体到下次怎么避免]
```

每次新会话启动,自动读取 `.claude/feedback.md` 最近 10 条作为上下文参考。
某条教训反复出现 ≥3 次,主动提醒用户考虑提炼为 CLAUDE.md 新规约项。