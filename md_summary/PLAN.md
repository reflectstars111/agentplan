# Agent-OS MVP 下一步完善计划

## Summary
下一步不直接进入多 Agent 并发，而是先完成“可验证 MVP 基线”：明确任务集、冻结最小接口、实现单 Agent + 多级记忆 + Context MMU 的纵向闭环。核心目标是先证明多级存储、混合检索、上下文装配、写回门控、Trace 是否真的优于普通 RAG。

## Key Changes
- 阶段 0：定义 5 个验收任务场景：文档问答、代码定位、项目连续问答、历史记忆辅助、冲突信息识别。
- 阶段 1：优先实现基础设施：Memory Store、File Store、Chunker、Vector/Keyword Hybrid Index、基础 Retriever。
- 阶段 2：实现 Context MMU 最小闭环：候选召回、去重、排序、token 预算、来源标注、Context Pack 输出。
- 阶段 3：接入单 Agent Runtime：Input Adapter -> Retriever -> Context MMU -> Worker -> Verifier -> Write-back Gate -> Trace Logger。
- 暂不实现：真实多 Agent 并发、复杂权限沙箱、完整知识图谱、完整 GUI、自主长期运行。

## Interfaces / Types
- 固定核心数据对象：`MemoryItem`、`DocumentChunk`、`ContextPack`、`ToolResult`、`TraceStep`。
- 固定最小表结构：`memories`、`chunks`、`traces`，`agents` 和 `tasks` 先保留 schema，不作为第一轮关键路径。
- Context Pack 必须包含：`task_id`、`budget`、`sections`、`source_refs`、`trust_level`。
- Write-back Gate 必须输出：`write/skip`、写入位置、置信度、是否需要用户确认、原因。
- Trace Logger 必须记录：输入、检索 query、候选结果、上下文装配、Verifier 结果、写回决策、最终输出来源。

## Test Plan
- 检索测试：每个任务场景准备标准问题和期望命中片段，评估 Top-k 命中率、无关 chunk 比例。
- 上下文测试：验证 Context MMU 能在固定 token 预算内保留关键证据，并过滤重复或低相关内容。
- 写回测试：验证重要结论会进入工作记忆，外部不可信内容不会污染长期记忆。
- 引用测试：Verifier 检查回答是否能追溯到真实 source reference。
- 对比测试：同一任务用普通 RAG 和 Agent-OS MVP 各跑一次，观察长任务连续性、来源准确性、trace 完整度。

## Assumptions
- 默认采用快速验证栈：Python + SQLite + FAISS/BM25 + PyMuPDF + FastAPI。
- 第一版以 CLI/API 原型为主，不做完整前端。
- 下一步交付物应先是 `MVP 需求与评估集文档`，然后再进入代码实现。
- 成功标准：至少完成 3 类任务闭环，并能证明 Context MMU、写回门控、Trace 对稳定性有可观察提升。
