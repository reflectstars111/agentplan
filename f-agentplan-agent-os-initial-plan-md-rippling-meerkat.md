# Agent-OS Phase 2: Controller + Task Graph 实施计划

## Context

Agent-OS MVP（阶段 0-1）已完成：20 次 TDD 提交，123 项全绿测试。系统当前是一个**单 Agent 问答管线**：上传文件 → 检索 → 上下文装配 → 推理 → 验证 → 写回 → 追踪。

依据 [agent_os_initial_plan.md](F:/agentplan/agent_os_initial_plan.md) §19 的 5 阶段路线图：

| 阶段 | 状态 |
|------|------|
| 0: 任务场景 + 评估集 | ✅ 已完成 |
| 1: 单 Agent + 多级记忆 | ✅ 已完成 |
| **2: Controller + Task Graph** | **⬅ 下一步** |
| 3: 多 Agent 协作 | 待定 |
| 4: 文件/代码深度索引 | 待定 |
| 5: 安全/权限/稳定 | 待定 |

**本计划的目标**：将系统从"问答"升级为"任务执行"——用户提出复杂需求，系统自动拆解为子任务并按依赖顺序执行。

---

## 架构决策

### 1. Controller 包装而非替换 AgentRuntime
- AgentRuntime 保持不变（零修改），单个任务节点的执行引擎
- Controller 作为外层编排器：Intent → Plan → Schedule → Execute
- 现有 `/query` 端点保留（简单模式），新增 `/task` 端点（任务模式）

### 2. Intent Decoder：关键词驱动（非 LLM）
- 为每种任务类型定义触发词集合
- 无需 LLM 调用，零延迟
- 后续可替换为 LLM 驱动的解码器（策略模式）

### 3. Planner：模板驱动（非 LLM）
- 3 个初始模板：document_qa、code_analysis、multi_turn
- 每个模板定义一个 2-5 节点的小型 DAG
- GENERAL 模板 = 单节点回退

### 4. Task Graph：邻接表 DAG（无外部依赖）
- 内存结构：nodes + adj_in + adj_out
- 插入时 DFS 环检测 + Kahn 拓扑排序
- 足够处理 MVP 规模（10-100 节点）

### 5. Scheduler：同步拓扑执行
- 单节点执行，无真实并发
- 失败重试 2 次 → 级联 SKIPPED 给下游节点
- 每个节点委托给 AgentRuntime.process_query()

---

## 新增文件（11 个）

```
src/
  models/
    intent.py              # Intent + IntentType
    task.py                # Task + TaskStatus + TaskGraph (DAG)
  runtime/
    intent_decoder.py      # IntentDecoder (keyword dispatch)
    planner.py             # Planner (template-based DAG construction)
    scheduler.py           # Scheduler (topological execution + retry)
    controller.py          # Controller (orchestrator)
  api/
    task_routes.py         # POST /task, GET /task/{id}/status

tests/
    test_intent.py         # Intent/Task 模型测试
    test_task_graph.py     # DAG 操作测试
    test_intent_decoder.py
    test_planner.py
    test_scheduler.py
    test_controller.py
    test_task_api.py
```

**零修改的现有文件**：`agent_runtime.py`、`verifier.py`、`writeback_gate.py`、`trace_logger.py`、`mmu.py`、`hybrid_retriever.py`、`storage/*`、`models/memory.py`、`models/context.py`、`db/connection.py`

**仅新增导出的文件**：`models/__init__.py`、`runtime/__init__.py`、`api/main.py`、`config.py`（+2 配置项）

---

## 核心数据模型

### Intent
```
IntentType: DOCUMENT_QA | CODE_ANALYSIS | MULTI_TURN | MEMORY_QUERY | GENERAL
Intent: intent_id, intent_type, original_query, entities[], constraints{}, priority (1-10), confidence (0-1)
```

### Task + TaskGraph
```
TaskStatus: CREATED → READY → RUNNING → COMPLETED / FAILED / SKIPPED
Task: task_id, task_type, agent_type="worker", status, dependencies[], input{}, output{}, retry_count, max_retries=2, error?, trace_id?

TaskGraph: nodes{}, adj_in{}, adj_out{}
  - add_node(), add_edge(), validate_acyclic() [DFS], topological_sort() [Kahn], get_ready_nodes()
```

---

## 组件接口

### IntentDecoder
```
decode(query: str) → Intent
  触发词调度：
    DOCUMENT_QA:  "what is", "explain", "how does", "summarize"
    CODE_ANALYSIS: "where is", "find", "which file", "locate"
    MEMORY_QUERY:  "what did we decide", "recall", "previous"
    MULTI_TURN:    上下文检测（工作记忆中有项目状态时）
    GENERAL:       回退
```

### Planner
```
plan(intent: Intent) → TaskGraph
  模板 DOCUMENT_QA (3 节点):
    [retrieve] → [reason] → [verify]
  模板 CODE_ANALYSIS (3 节点):
    [retrieve] → [analyze] → [verify]
  模板 MULTI_TURN (5 节点):
    [retrieve_memory] ─┐
                        ├→ [merge] → [reason] → [verify] → [writeback]
    [retrieve_chunks] ─┘
  模板 GENERAL (1 节点):
    [retrieve+reason]
```

### Scheduler
```
execute(task_graph: TaskGraph) → {results, status, trace_ids, failed_tasks}
  循环:
    1. get_ready_nodes(completed_set) → 所有依赖已满足的节点
    2. 对每个就绪节点: _execute_node(task) → AgentRuntime.process_query()
    3. 成功 → completed, 失败 → retry 或 FAILED → 下游 SKIPPED
    4. 直到 all_completed() 或死锁
```

### Controller
```
process(query: str) → {response, intent, task_graph_summary, results, status, trace_ids}
  流程: IntentDecode → Plan → Schedule → Execute → AssembleResponse
  
process_query(query: str) → {response, trace_id, ...}
  向后兼容：直接委托给 AgentRuntime.process_query()
```

---

## TDD 任务分解（12 个任务）

### 第一批（可并行）：模型 + 基础
| # | 任务 | 测试文件 | 新增文件 |
|---|------|---------|---------|
| 1 | Intent + Task 数据模型 + TaskGraph DAG | test_intent.py, test_task_graph.py | models/intent.py, models/task.py |
| 2 | IntentDecoder（关键词调度 + 实体提取） | test_intent_decoder.py | runtime/intent_decoder.py |
| 3 | Planner（模板注册 + DAG 构建） | test_planner.py | runtime/planner.py |

### 第二批：调度器
| # | 任务 | 测试文件 | 新增文件 |
|---|------|---------|---------|
| 4 | Scheduler（拓扑执行 + 重试 + 级联失败） | test_scheduler.py | runtime/scheduler.py |

### 第三批：编排器 + 集成
| # | 任务 | 测试文件 | 新增文件 |
|---|------|---------|---------|
| 5 | Controller（编排器，包装 AgentRuntime） | test_controller.py | runtime/controller.py |
| 6 | INTENT_DECODE 追踪步骤 + 状态报告 | test_controller.py（扩展） | — |
| 7 | 配置项：task_max_retries、task_default_priority | — | config.py |

### 第四批：API + E2E
| # | 任务 | 测试文件 | 新增文件 |
|---|------|---------|---------|
| 8 | POST /task + GET /task/{id}/status 端点 | test_task_api.py | api/task_routes.py |
| 9 | 向后兼容：/query 端点不受影响 | test_api.py（验证） | — |
| 10 | E2E：任务模式处理文档问答场景 | test_e2e_scenarios.py（扩展） | — |
| 11 | E2E：任务模式处理多轮项目连续性 | test_e2e_scenarios.py（扩展） | — |
| 12 | 全部测试回归：≥135 项测试全绿 | — | — |

---

## 迁移策略：零破坏性变更

```
Phase 2a (Tasks 1-3): 仅新建 models + IntentDecoder + Planner
  ├─ 运行全量测试 → 123 项通过 + N 项新测试通过
  └─ 门控: 零回归

Phase 2b (Task 4): 新建 Scheduler
  ├─ Scheduler 依赖 AgentRuntime 的公开 API，不修改内部
  └─ 门控: 零回归

Phase 2c (Tasks 5-7): 新建 Controller + 配置
  ├─ Controller 包装 AgentRuntime，不修改
  └─ 门控: 零回归

Phase 2d (Tasks 8-12): API + E2E
  ├─ 新增 /task 路由，不修改 /query 路由
  ├─ 在 main.py 中增量注册（app.include_router）
  └─ 门控: ≥135 项测试全绿
```

---

## 模板示例：DOCUMENT_QA 执行流程

```
用户输入: "What is the main contribution of this paper?"
    │
    ▼
IntentDecoder.decode()
    → Intent(intent_type=DOCUMENT_QA, entities=["paper"], confidence=0.9)
    │
    ▼
Planner.plan(intent)
    → TaskGraph: [retrieve] → [reason] → [verify]
    │
    ▼
Scheduler.execute(task_graph)
    │
    ├─ Node "retrieve": AgentRuntime.process_query("retrieve relevant chunks about paper")
    │   → {response: "...", chunks: [...]}
    │
    ├─ Node "reason": AgentRuntime.process_query("What is the main contribution? Context: [chunks]")
    │   → {response: "The main contribution is RAPTOR..."}
    │
    └─ Node "verify": Verifier.verify("The main contribution is RAPTOR...", context_pack)
        → {is_verified: True, unverified_claims: []}
    │
    ▼
Controller 返回:
    {
      response: "The main contribution is RAPTOR...",
      intent: {type: DOCUMENT_QA, ...},
      task_graph_summary: {node_count: 3, completed: 3, failed: 0},
      status: "completed",
      trace_ids: ["trace_abc", "trace_def", "trace_ghi"],
    }
```

---

## 验证计划

### 单元测试验证
- `python -m pytest tests/test_intent.py tests/test_task_graph.py -v` — 模型正确性
- `python -m pytest tests/test_intent_decoder.py -v` — 意图识别准确率
- `python -m pytest tests/test_planner.py -v` — 模板生成 DAG 结构正确
- `python -m pytest tests/test_scheduler.py -v` — 拓扑执行 + 重试 + 级联

### 集成测试验证
- `python -m pytest tests/test_controller.py -v` — 编排器完整流程
- `python -m pytest tests/test_task_api.py -v` — HTTP 端点

### 回归测试验证
- `python -m pytest tests/ -v` — 全部 ≥135 项测试通过

### 手动验证
- 启动 `uvicorn src.api.main:create_app --factory` 后：
  - `POST /query` 返回和之前相同的响应格式
  - `POST /task` 返回包含 intent + task_graph_summary 的任务模式响应
  - `GET /health` 返回 `{"status": "ok"}`
