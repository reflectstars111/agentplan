# Agent Runtime Database (ARD)

## 一、项目重新定位

### 核心定义
Agent Runtime Database（ARD）是一种以状态管理（State Management）为核心的 Agent 运行时架构。

目标不是构建一个“Agent 操作系统”，而是构建一个能够管理长期状态、知识、上下文与执行过程的 Runtime Database。

### 核心研究问题

#### RQ1
如何在有限 Context Window 下稳定调用长期知识与历史状态？

#### RQ2
如何管理 Agent 执行过程中不断增长的状态与知识？

#### RQ3
如何保证记忆（状态）写入的一致性、可验证性与可追溯性？

---

# 二、核心思想

ARD 融合三类思想：

- Operating System
- Database System
- Retrieval-Augmented Generation

其中数据库思想优先级最高。

Agent 面临的核心问题本质是：

- State Explosion
- Knowledge Explosion
- Trace Explosion

因此系统重点不再是 Agent 数量，而是：

- State Management
- Context Management
- Knowledge Management

---

# 三、总体架构

```text
                 User
                   │
                   ▼

            Task Planner
                   │
                   ▼

            Query Planner
                   │
                   ▼

          Context MMU
                   │

      ┌────────────┼────────────┐
      ▼            ▼            ▼

 State Store  Knowledge Store  Trace Store

      ▲            ▲            ▲

      └────────────┼────────────┘
                   │

           Agent Executor
                   │

            Tool Router
                   │

              Verifier
                   │

        Transaction Manager
                   │

               Commit
```

---

# 四、分层存储体系

## L0 Context Buffer

当前 Prompt 与当前输入。

## L1 Session Store

最近对话历史。

## L2 Working State Store

保存：

- 当前任务状态
- Agent 状态
- 中间推理结果
- 执行进度

## L3 Long-Term Knowledge Store

保存：

- 项目知识
- 用户偏好
- 历史决策
- 长期事实

## L4 External Knowledge Store

保存：

- PDF
- GitHub Repository
- Web
- Database
- Dataset

## L5 Archive Store

保存：

- 历史版本
- 废弃决策
- 冷数据

---

# 五、State 替代 Memory

不再使用 Memory 作为核心抽象。

统一使用：

```text
State
```

State 包含：

- User State
- Project State
- Agent State
- Task State
- Knowledge State

这样更符合系统运行时本质。

---

# 六、Context MMU（核心创新）

Context MMU 是系统核心模块。

职责：

1. Retrieve
2. Filter
3. Rank
4. Compress
5. Assemble
6. Budget Allocation

流程：

```text
User Query
      ↓

Query Planner

      ↓

Hybrid Retrieval

      ↓

Context MMU

      ↓

Context Package

      ↓

LLM
```

## Context Page Fault

当推理所需信息不存在于当前上下文时：

```text
Page Fault
↓
Generate Query
↓
Retrieve
↓
Load Context
↓
Continue
```

---

# 七、Hybrid Retrieval

系统采用混合检索。

综合：

- Vector Retrieval
- Keyword Retrieval
- Structural Retrieval
- Temporal Retrieval
- Graph Retrieval

推荐评分函数：

```text
Score =
0.35 Semantic
+0.20 Keyword
+0.15 Entity
+0.10 Recency
+0.10 Importance
+0.10 Structure
-0.10 TokenCost
-0.20 TrustPenalty
```

---

# 八、事务化状态写回

引入数据库事务机制。

## State Transaction

```json
{
  "txn_id":"txn_001",
  "read_set":[],
  "write_set":[],
  "status":"pending"
}
```

执行流程：

```text
BEGIN
↓
READ
↓
REASON
↓
VERIFY
↓
COMMIT
```

失败：

```text
ROLLBACK
```

---

# 九、MVCC 版本管理

采用 Multi-Version Concurrency Control。

示例：

```text
Decision_v1
Decision_v2
Decision_v3
```

而不是覆盖旧状态。

优势：

- 保留历史设计过程
- 支持追溯
- 支持审计
- 支持冲突恢复

---

# 十、Trace Store

记录完整执行链路。

包括：

- User Request
- Retrieval
- Context Assembly
- Tool Call
- Verification
- State Commit

Trace 成为系统第三大存储。

---

# 十一、Agent 设计原则

弱化 Agent。

MVP 仅保留：

## Planner

任务拆解。

## Executor

执行任务。

## Verifier

验证结果。

避免过早引入复杂多 Agent 协作。

---

# 十二、Graph Store（扩展方向）

建立实体关系图。

例如：

```text
Project
 ├─ Paper
 ├─ Code
 ├─ Decision
 └─ Dataset
```

支持：

- Graph Retrieval
- Dependency Analysis
- Knowledge Navigation

---

# 十三、MVP 实现范围

必须实现：

1. State Store
2. Knowledge Store
3. Trace Store
4. Hybrid Retriever
5. Context MMU
6. Planner
7. Executor
8. Verifier
9. Transaction Manager

可选：

10. Graph Store

---

# 十四、开发路线图

Phase 1

- State Store
- Knowledge Store
- File Store
- Chunking
- Indexing

Phase 2

- Hybrid Retrieval
- Context MMU
- Context Page Fault

Phase 3

- Planner
- Executor
- Verifier

Phase 4

- Transaction Manager
- MVCC
- Trace Store

Phase 5

- Graph Store
- Multi-Agent Collaboration

---

# 十五、项目最终定义

Agent Runtime Database（ARD）是一种面向长期复杂任务的状态中心型 Agent 运行时架构。

其核心贡献包括：

1. Hierarchical State Architecture
2. Context MMU
3. Context Page Fault Retrieval
4. Transactional State Write-back
5. MVCC Memory Evolution
6. Traceable Runtime Execution

相比传统 Agent Framework：

- 更强调状态管理
- 更强调知识管理
- 更强调可追溯性
- 更强调长期任务连续性

目标是在有限 Context Window 下实现稳定、可验证、可扩展的长期 Agent Runtime。
