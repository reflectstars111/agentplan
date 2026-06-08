# P0 接线修复 — 设计文档

日期: 2026-06-07 | 状态: 待实现

## 背景

对 `agent_os_initial_plan.md` 与代码实现的 gap 分析识别了 20 个差异点。P0 是其中的 4 个严重问题：已实现的模块未接入主执行回路，或数据不一致导致静默丢失。

## P0-1: ContextPageFault 接入 Controller/Scheduler

### 问题

`src/context/page_fault.py` 完整实现了不确定性检测 → 缺失信息提取 → 二次检索 → 更新 ContextPack 的流程，但从未被调用。

### 设计

在 `Scheduler._execute_one()` 中，task 执行后触发：

```
task 执行 → 响应
  → ContextPageFault._needs_more_context(response) ?
    YES → page_fault.check_and_handle(response, context_pack, query, embed_fn)
      → triggered? → 用 updated_pack 重新调用 LLM
      → not triggered? → 正常返回
    NO → 正常返回
```

**改动文件:**
- `src/runtime/scheduler.py` — `__init__` 新增可选参数 `page_fault: ContextPageFault | None`；`_execute_one()` 在 LLM 推理后加入 page fault 检测和重试循环
- `src/runtime/agent_runtime.py` — `_step_reason()` 需要额外的 `embed_fn` 和 `page_fault` 参数支持重试

**约束:** 每个 task 最多 2 次 page fault（`ContextPageFault.max_faults=2`，调用前 `reset()` 每次 task 重置计数器）。

**不涉及:** API 层、前端。

---

## P0-2: DB Schema 补 `last_used_at` 列

### 问题

`MemoryItem` 模型定义了 `last_used_at: datetime | None` 字段（用于 HybridRetriever 时间索引评分），但 `memories` 表 DDL 中没有此列 — 写入时静默丢失。

### 设计

**DDL 变更:**
- `MEMORIES_TABLE` 新增 `last_used_at TEXT` 列（DEFAULT NULL，向后兼容）

**MemoryStore 变更:**
- `insert()` — 写入 `last_used_at`
- `_row_to_item()` — 读取 `last_used_at`
- 新增 `touch(memory_id: str)` — 更新 `last_used_at` 为当前时间

**HybridRetriever 变更:**
- `_recency_score()` — 优先使用 `last_used_at`，回退到 `created_at`

**MemoryItem 模型:** 无需改动（字段已存在）。

**不涉及:** `already_active` 的现有记忆记录（`last_used_at` 为 NULL 时 recency_score 返回默认值 0.5）。

---

## P0-3: Reranker 集成进 HybridRetriever

### 问题

`src/index/reranker.py` 实现了精确匹配加分、来源多样性、长度惩罚等重排序逻辑，但检索链路（`AgentRuntime._step_retrieve()`）不使用它。

### 设计

**HybridRetriever 变更:**
- 构造器新增可选参数 `reranker: Reranker | None = None`
- 新增方法 `retrieve_and_rerank(query, embed_fn, k=10, filters=None)`:
  1. 调 `self.retrieve(query, embed_fn, k * 2, filters)` — 取 2 倍候选
  2. 调 `self.reranker.rerank(results, query, top_k=k)` — 精排取 top_k
  3. 若 `reranker is None`，回退到 `self.retrieve()`

**AgentRuntime 变更:**
- `_step_retrieve()` 改为 `self.retriever.retrieve_and_rerank(...)`

**向后兼容:** `retrieve()` 原方法不变。无 reranker 时行为不变。

**不涉及:** `QueryPlanner`, `Controller`.

---

## P0-4: Controller 全步骤 Trace 记录

### 问题

`Controller.process()` 只记录 IntentDecode 和 Respond 两个 step。Plan、Schedule、Execute 阶段的 step 缺失 — 执行链路在 trace 中断裂。

### 设计

**Controller.process() 补入:**
- Plan 步骤: `StepType.PLAN`（需在 `StepType` 枚举中新增）
- Schedule 步骤: `StepType.SCHEDULE`（需在 `StepType` 枚举中新增）

**Scheduler.execute() 补入:**
- 每个 task 执行后记录 `StepType.LLM_REASONING` step（需要 `trace_logger` 依赖）

**改动文件:**
- `src/models/trace.py` — `StepType` 枚举新增 `PLAN`, `SCHEDULE`
- `src/runtime/controller.py` — `process()` 中补 Plan 和 Schedule 的 step 记录
- `src/runtime/scheduler.py` — `__init__` 新增可选 `trace_logger`；`_execute_one()` 记录每个 task 的 step

**不涉及:** `AgentRuntime`（已有完整 trace）。

---

## 不涉及的内容

以下不在 P0 范围内，保留到后续轮次：

- P1: EntityIndex/DependencyGraph 接入上传、ConversationCache 接入 API、全链路 PageFault
- P2: Word/DB/API/图片输入、L5 冷归档、输出格式化、InterruptHandler 接入
- P3: Agent 状态机驱动、语义压缩、SPAWN_AGENT 指令

## 测试策略

每项修复配对应的单元测试：
1. **PageFault**: `test_scheduler_page_fault.py` — mock 一个返回不确定标记的 LLM，验证重检索和重试
2. **DB last_used_at**: 修改 `test_memory_store.py` — 验证字段读写
3. **Reranker**: `test_hybrid_retriever.py` 新增 `test_retrieve_and_rerank`
4. **Trace**: `test_controller.py` 验证 trace step 数量 ≥ 4

## 文件变更清单

| 文件 | 变更类型 |
|---|---|
| `src/runtime/scheduler.py` | P0-1: +page_fault 参数, +重试循环; P0-4: +trace_logger |
| `src/runtime/agent_runtime.py` | P0-1: _step_reason 支持 page_fault 重入 |
| `src/db/migrations.py` | P0-2: MEMORIES_TABLE +last_used_at |
| `src/storage/memory_store.py` | P0-2: +touch(), insert/read 适配 |
| `src/index/hybrid_retriever.py` | P0-3: +reranker, +retrieve_and_rerank() |
| `src/models/trace.py` | P0-4: StepType +PLAN, +SCHEDULE |
| `src/runtime/controller.py` | P0-4: +plan/schedule trace steps |
| `tests/test_scheduler_page_fault.py` | 新增 |
| `tests/test_memory_store.py` | 扩展 |
| `tests/test_hybrid_retriever.py` | 扩展 |
| `tests/test_controller.py` | 扩展 |
