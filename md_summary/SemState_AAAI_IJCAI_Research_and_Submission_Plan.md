# SemState：冲击 AAAI / IJCAI 的研究与投稿执行计划

> **项目基础**：Agent Runtime Database（ARD）  
> **目标方向**：多智能体系统、LLM Agent 可靠性、共享状态协调、冲突检测与计划修复  
> **目标会议**：AAAI、IJCAI  
> **计划版本**：v1.0  
> **制定日期**：2026-06-09

---

## 0. 文档目的

本文档将现有 Agent-OS、ARD 论文与 SemState 转型方案统一为一条面向 AAAI / IJCAI 的主论文路线，回答以下问题：

1. 现有 ARD 工作中哪些内容继续复用；
2. 主论文应研究什么问题，而不应继续研究什么问题；
3. SemState 需要提出哪些真正具有 AI 研究价值的方法；
4. 如何构建具有客观 Ground Truth 的多 Agent 状态一致性 Benchmark；
5. 如何设计基线、指标、消融实验和错误分析；
6. 如何在 AAAI-27 截止日前完成最低可投稿版本；
7. 如何将 AAAI 版本继续增强为 IJCAI 版本。

---

# 1. 执行摘要

## 1.1 核心决策

后续主论文不再以以下表述为中心：

> 我们提出了一个基于 Event Store、Context MMU、OCC 和版本查询的通用 Agent Runtime。

推荐改为：

> 多个 LLM Agent 即使没有产生同键写冲突，也可能由于过期依赖、隐含假设不兼容和派生产物失效，形成结构上合法但语义上错误的共享状态。我们提出 SemState，通过自动依赖发现、混合语义验证、失效传播和最小子图修复，提高多 Agent 长程任务的最终状态正确性。

## 1.2 最终论文形态

主论文由四部分构成：

1. **新问题**：Semantic State Conflict；
2. **新方法**：自动依赖发现与混合语义验证；
3. **新恢复机制**：Dependency-aware Targeted Repair；
4. **新 Benchmark**：具有客观 Ground Truth 的多领域、多调度共享状态一致性基准。

## 1.3 ARD 与 SemState 的关系

```text
Agent-OS 初始构想
        ↓
ARD 状态运行时
        ↓
Versioned Shared State
        ↓
Structural OCC
        ↓
SemState
        ↓
Semantic Conflict Detection
        ↓
Dependency Invalidation
        ↓
Targeted Repair
```

ARD 不被放弃，而是调整为：

> **An experimental runtime substrate for studying semantic consistency in versioned multi-agent state.**

即：

- ARD 是底层运行时；
- SemState 是主论文的新方法；
- ARD 原有实验是预实验和组件验证；
- SemState Benchmark 和端到端任务实验是主论文证据。

---

# 2. 为什么必须从 ARD 转向 SemState

## 2.1 ARD 已有价值

现有系统已经具备：

- append-only Event Store；
- versioned StateStore；
- TransactionManager；
- optimistic concurrency control；
- read-set version validation；
- point-in-time state query；
- State history；
- TraceStore；
- provenance/source references；
- ContextPack；
- Planner、Executor、Verifier 的基础运行链；
- provenance、persistent state、context capacity 的初步实验。

这些内容足以构成新方法的实验底座，无需推倒重写。

## 2.2 ARD 作为主论文的不足

当前 ARD 论文同时讨论：

- 通用 Agent Runtime；
- Context MMU；
- provenance；
- persistent state；
- context length；
- event sourcing；
- OCC；
- version query；
- 审计和恢复。

问题在于：

1. 研究主张过多；
2. 各部分证据强弱不一致；
3. provenance 的证据最强，但与多 Agent 协调主线不完全统一；
4. persistent state 的效果具有明显场景差异；
5. context length 实验仍属于小规模 pilot；
6. OCC、版本恢复属于成熟机制的应用级实现；
7. 容易被审稿人评价为“系统功能集合”而不是一个清晰的新 AI 问题。

## 2.3 SemState 的研究缺口

传统 OCC 能发现：

- 同一个 key 被并发修改；
- Agent 基于旧版本读取后提交；
- lost update；
- 显式 read-set stale read。

但是它不能充分发现：

```text
Agent A:
database_engine = PostgreSQL

Agent B:
migration_script = MySQL dialect
```

两个 Agent 修改不同字段，没有普通 write-write conflict，但最终项目状态不可执行。

因此，主论文研究的问题应从：

> 如何保存和版本化 Agent 状态？

推进为：

> 如何判断多个 Agent 对相互依赖状态的更新在语义上是否兼容，并在异常发生后进行最小代价修复？

---

# 3. 论文定位与题目建议

## 3.1 推荐主标题

### 方案 A：问题驱动，最适合 AAAI / IJCAI

**When Version Checks Are Not Enough: Detecting and Repairing Semantic State Conflicts in Multi-Agent LLM Systems**

### 方案 B：系统名驱动

**SemState: Semantic State Consistency for Multi-Agent LLM Workflows**

### 方案 C：强调依赖与修复

**SemState: Dependency-Aware Validation and Targeted Repair for Shared State in LLM Agent Workflows**

## 3.2 推荐一句话定位

> SemState studies a class of multi-agent failures where concurrent updates are structurally valid but semantically inconsistent, and introduces trace-derived dependency discovery, hybrid validation, and targeted workflow repair.

## 3.3 AAAI / IJCAI 的学术归属

论文应被定位为：

- Multi-Agent Systems；
- Coordination and Collaboration；
- LLM Agent Reliability；
- Distributed Problem Solving；
- Replanning and Plan Repair；
- Knowledge Representation and Reasoning。

不建议把论文主要定位为：

- 数据库事务系统；
- RAG；
- 长上下文；
- 通用 Agent Framework；
- 纯 Agent Memory。

---

# 4. 核心研究问题与假设

## RQ1：普通版本检查遗漏了哪些语义状态冲突？

研究：

- 同键冲突与跨键语义冲突的差异；
- 不同 Agent 数量和调度顺序下异常发生率；
- 传统 OCC、read-set OCC 对不同异常类型的覆盖范围。

### 假设 H1

> Per-key version checks and read-set OCC reduce structural conflicts but cannot adequately prevent cross-key semantic inconsistencies.

---

## RQ2：自动依赖发现和混合验证能否提高语义冲突检测能力？

研究：

- 显式依赖；
- trace-derived dependency；
- schema/graph dependency；
- latent semantic dependency；
- deterministic validator 与 LLM validator 的组合。

### 假设 H2

> Combining runtime traces, structural dependencies, provenance, and semantic constraints yields higher conflict recall than version-only and explicit-dependency baselines.

---

## RQ3：依赖感知失效传播能否准确识别受影响状态？

研究：

- 上游状态变化后的下游影响范围；
- stale、invalid、needs_verification 的状态分类；
- 漏标记与过度标记；
- provenance 对根因定位的作用。

### 假设 H3

> Dependency-aware invalidation identifies affected artifacts more accurately than global invalidation or local key-level invalidation.

---

## RQ4：Targeted Repair 能否以较低成本恢复任务正确性？

研究：

- 全量重跑；
- 局部重跑；
- 最小受影响子图；
- 重试次数；
- 修复后任务成功率；
- token、LLM call 和延迟成本。

### 假设 H4

> Targeted repair achieves task success comparable to full workflow re-execution while using substantially fewer model calls and tokens.

---

## RQ5：依赖信息不完整时，SemState 是否仍然稳健？

研究：

- 删除真实依赖；
- 添加错误依赖；
- Agent 漏报读取；
- provenance 缺失；
- 隐含依赖推断错误。

### 假设 H5

> SemState degrades gracefully under incomplete dependency information and remains more reliable than version-only coordination over a practical range of dependency noise.

---

# 5. 建议的三项核心贡献

正文只能集中讲三项方法贡献，避免再次扩张。

## 贡献一：Semantic State Conflict Problem and Benchmark

定义“结构上合法、语义上不一致”的多 Agent 共享状态异常，提出：

- 状态表示；
- 事务表示；
- 异常 taxonomy；
- 最终状态有效性；
- SemStateBench。

## 贡献二：Automatic Dependency Discovery and Hybrid Validation

从以下来源自动构建依赖：

1. Agent 实际读取；
2. ContextPack 中加载的状态；
3. 工具调用输入输出；
4. source references；
5. typed state graph；
6. 输出内容中的隐含语义依赖。

并结合：

- version validation；
- schema validation；
- deterministic constraints；
- executable tests；
- provenance validation；
- semantic contradiction detection。

## 贡献三：Dependency-Aware Targeted Repair

检测冲突后：

1. 定位根因；
2. 传播状态失效；
3. 计算受影响任务子图；
4. 只重跑必要 Agent；
5. 重建新版本；
6. 保留完整 repair trace。

---

# 6. 问题形式化

## 6.1 Versioned Shared State Graph

将共享状态表示为：

\[
G=(V,E_d,E_p)
\]

其中：

- \(V\)：版本化状态节点；
- \(E_d\)：状态依赖边；
- \(E_p\)：证据和 provenance 边。

状态节点示例：

```json
{
  "key": "migration_plan",
  "value": "...",
  "type": "derived_artifact",
  "version": 4,
  "status": "active",
  "read_dependencies": [
    {"key": "database_engine", "version": 2},
    {"key": "data_schema", "version": 5}
  ],
  "source_refs": [
    "decision_023",
    "schema_doc_v3"
  ],
  "created_by": "agent_migration",
  "task_id": "task_017",
  "derived_from": [
    "database_engine@v2",
    "data_schema@v5"
  ],
  "supersedes": "migration_plan@v3"
}
```

## 6.2 Agent Transaction Envelope

```json
{
  "txn_id": "txn_017",
  "agent_id": "agent_test",
  "task_id": "task_017",
  "read_set": {
    "database_engine": 2,
    "api_contract": 5
  },
  "write_set": {
    "test_plan": "..."
  },
  "dependency_set": [
    "database_engine",
    "api_contract"
  ],
  "evidence_refs": [
    "schema_doc_v3",
    "api_spec_v5"
  ],
  "tool_trace_refs": [
    "tool_call_102"
  ]
}
```

## 6.3 轻量语义一致性定义

AAAI / IJCAI 版本不需要发展成完整数据库隔离理论，但必须给出明确条件：

> 一个多 Agent 执行历史是语义一致的，当其最终共享状态满足所有版本依赖、状态依赖、证据依赖和领域约束，并且不存在仍被标记为 active 的过期派生产物。

可进一步定义：

- Structural validity；
- Dependency validity；
- Evidence validity；
- Constraint validity；
- Workflow validity。

---

# 7. 语义异常 Taxonomy

建议正文保留 5–6 类，附录扩展。

## A1. Same-Key Lost Update

两个 Agent 修改相同状态，后写覆盖前写。

用途：验证传统 OCC 的基本能力，不是主要创新。

## A2. Cross-Key Stale Dependency

上游状态已更新，另一个 Agent 仍基于旧版本生成不同 key。

```text
database_engine@v2 → PostgreSQL
migration_plan based on database_engine@v1 → MySQL script
```

## A3. Derived Artifact Staleness

上游决策改变，但派生产物仍为 active。

```text
api_contract changed
test_plan based on old api_contract remains active
```

## A4. Conflicting Agent Assumptions

两个 Agent 写入不同字段，但使用了互相不兼容的隐含假设。

```text
architecture = serverless
deployment_plan = long-running local daemon
```

## A5. Evidence-Version Mismatch

状态更新了，但引用仍指向旧证据或已经被替代的文档版本。

## A6. Partial Semantic Commit

一个任务本应原子更新多个相关状态，但只有部分结果提交成功。

## A7. Missing Dependency Invalidation

上游状态改变后，下游节点未被标记 stale 或 needs_verification。

## A8. Invalid Repair

修复过程重跑了错误任务、漏掉关键任务，或重新引入冲突。

---

# 8. SemState 方法设计

## 8.1 总体流程

```text
Agent Execution
      ↓
Transaction Envelope
      ↓
Dependency Collector
      ↓
Structural Validator
      ↓
Dependency Validator
      ↓
Semantic Constraint Checker
      ↓
Commit / Reject / Mark Uncertain
      ↓
Dependency Invalidation
      ↓
Targeted Repair Planner
      ↓
Selective Agent Re-execution
      ↓
Versioned Commit + Repair Trace
```

---

## 8.2 自动依赖发现

### 层 1：Observable Dependency

从运行轨迹确定性获取：

- StateStore read；
- ContextPack items；
- Tool inputs；
- Tool outputs；
- source refs；
- prior task outputs；
- Agent message references。

### 层 2：Structural Dependency

来自：

- typed schema；
- 人工定义的领域 DAG；
- workflow DAG；
- code/data lineage；
- task input/output declaration。

### 层 3：Latent Semantic Dependency

使用以下方法推断：

- 规则和关键词；
- schema slot matching；
- NLI/contradiction model；
- LLM dependency classifier；
- consistency verifier。

### 输出

```json
{
  "target": "deployment_plan",
  "dependencies": [
    {
      "key": "architecture",
      "source": "trace",
      "confidence": 1.0
    },
    {
      "key": "cloud_provider",
      "source": "semantic_inference",
      "confidence": 0.82
    }
  ]
}
```

---

## 8.3 混合语义验证

验证顺序必须优先使用客观工具：

1. **Version checker**
2. **Schema/type checker**
3. **Dependency version checker**
4. **Evidence-version checker**
5. **Executable validator**
6. **Domain rules**
7. **NLI/semantic verifier**
8. **LLM verifier**

原则：

> 能确定性验证的内容，不交给 LLM Judge。

输出应包含：

```json
{
  "decision": "reject",
  "conflict_type": "cross_key_stale_dependency",
  "root_state": "database_engine@v2",
  "affected_state": "migration_plan@v4",
  "evidence": [
    "migration_plan derived from database_engine@v1"
  ],
  "confidence": 1.0
}
```

---

## 8.4 失效传播

当上游节点更新时，下游状态可被标记为：

- `stale`；
- `invalid`；
- `needs_verification`；
- `potentially_stale`；
- `active`。

传播规则应综合：

- 边类型；
- 依赖强度；
- 版本变化类型；
- 是否存在可执行验证器；
- 状态重建成本。

---

## 8.5 Targeted Repair

修复目标：

\[
R^*=\arg\min_{R\subseteq T}
\sum_{t\in R} cost(t)
\]

满足：

- 所有 invalid 状态被重新生成或废弃；
- 所有 hard constraints 成立；
- 所有 active 状态依赖当前有效版本；
- 最终工作流目标完成。

修复算法：

1. 找到异常根因；
2. 计算受影响状态闭包；
3. 将状态节点映射到生成任务；
4. 去除已通过重新验证的节点；
5. 按任务依赖进行拓扑排序；
6. 重新加载最新状态；
7. 重跑最小任务集合；
8. 验证并提交新版本。

---

# 9. Benchmark 设计

## 9.1 Benchmark 名称

推荐：

- **SemStateBench**
- **MASS-Consistency**
- **AgentStateBench**

优先推荐 **SemStateBench**，与方法名一致。

## 9.2 任务领域

### D1. 软件系统设计

状态：

- architecture；
- database engine；
- data schema；
- API contract；
- migration；
- tests；
- deployment。

客观验证：

- schema validation；
- API tests；
- SQL parser；
- configuration rules；
- code execution。

### D2. 数据库迁移

状态：

- source database；
- target database；
- schema mapping；
- migration scripts；
- rollback；
- validation results。

客观验证：

- SQL dialect parser；
- schema diff；
- migration execution；
- row-level checks。

### D3. 科研报告协作

状态：

- research question；
- hypothesis；
- dataset；
- experimental setting；
- results；
- conclusion；
- citations。

验证：

- dataset/setting alignment；
- numerical result checks；
- citation-version consistency；
- conclusion-result entailment；
- 人工双重标注。

### D4. Incident Response

状态：

- incident cause；
- service status；
- temporary mitigation；
- user notification；
- permanent fix；
- postmortem。

验证：

- timeline rules；
- service dependency；
- causal consistency；
- action-status alignment。

### 可选 D5. 数据分析流水线

状态：

- data version；
- preprocessing；
- feature set；
- model；
- metrics；
- interpretation。

### 可选 D6. 基础设施部署

状态：

- cloud provider；
- runtime；
- network；
- permissions；
- deployment；
- monitoring。

---

## 9.3 单个场景格式

每个场景必须包含：

```json
{
  "scenario_id": "software_001",
  "initial_state": {},
  "agents": [],
  "task_dag": {},
  "state_dependency_graph": {},
  "constraints": [],
  "evidence_versions": [],
  "schedules": [],
  "valid_serial_outcomes": [],
  "expected_anomalies": [],
  "validators": []
}
```

## 9.4 并发 Schedule

至少包含：

```text
S1: A read → A write → B read → B write
S2: A read → B read → A write → B write
S3: A read old state → B update → A commit
S4: A update parent → B commit stale child
S5: A/B update different keys with conflicting assumptions
S6: upstream update → missing downstream invalidation
S7: conflict detected → incomplete repair
```

## 9.5 Ground Truth 来源

优先级：

1. 程序执行；
2. 编译器/parser；
3. unit tests；
4. schema validator；
5. deterministic constraints；
6. task-specific evaluator；
7. 人工双重标注；
8. LLM Judge 仅作为补充。

## 9.6 推荐规模

### AAAI-27 最低投稿版本

| 项目 | 目标 |
|---|---:|
| 领域 | 4 |
| 基础任务 | 60–100 |
| 每任务 schedule | 5–8 |
| 总执行历史 | 500–800 |
| 异常类型 | 5–6 |
| Agent 数量 | 2、4、6 |
| 模型系列 | 至少 3 |
| 随机重复 | 3 次以上 |

### IJCAI 增强版本

| 项目 | 目标 |
|---|---:|
| 领域 | 5–6 |
| 基础任务 | 150–250 |
| 总执行历史 | 1,500–3,000 |
| 模型系列 | 4–6 |
| Agent 数量 | 2、4、6、8 |
| 随机重复 | 5 次以上 |
| 人工标注子集 | 200+ |

---

# 10. 实验基线

## 10.1 无协调基线

- No Coordination；
- Last Write Wins；
- Shared Memory without Validation。

## 10.2 串行化基线

- Fully Sequential；
- Global Lock；
- Per-key Lock。

## 10.3 版本协调基线

- Per-key Version Check；
- Basic OCC；
- Read-set OCC；
- Dependency-aware OCC with Oracle Dependencies。

## 10.4 Agent 验证基线

- Agent Self-Reflection；
- Independent Verifier Agent；
- Debate/Consensus Verifier；
- Full Workflow Rerun。

## 10.5 SemState 变体

- SemState-Version；
- SemState-ExplicitDep；
- SemState-TraceDep；
- SemState-SemanticDep；
- SemState-Detect；
- SemState-Detect+Invalidate；
- SemState-FullRepair。

---

# 11. 评价指标

## 11.1 冲突检测

- Semantic Conflict Precision；
- Semantic Conflict Recall；
- F1；
- False Rejection Rate；
- Invalid Commit Rate；
- Conflict Type Accuracy。

## 11.2 最终状态和任务

- Final State Correctness；
- Constraint Satisfaction Rate；
- End-to-End Task Success；
- Cross-Agent Consistency；
- Stale Artifact Rate；
- Evidence-Version Consistency。

## 11.3 依赖发现

- Dependency Precision；
- Dependency Recall；
- Edge F1；
- Hidden Dependency Recall；
- Dependency Calibration。

## 11.4 失效传播

- Invalidation Precision；
- Invalidation Recall；
- Over-invalidation Rate；
- Missed Invalidation Rate。

## 11.5 修复

- Repair Success Rate；
- Minimal Re-execution Ratio；
- Unnecessary Rerun Rate；
- Reintroduced Error Rate；
- Retry Count；
- Time to Consistency。

## 11.6 成本

- LLM Calls；
- Input Tokens；
- Output Tokens；
- Latency；
- p50/p95 Commit Latency；
- Verifier Overhead；
- Repair Cost；
- Total Workflow Cost。

---

# 12. 主实验矩阵

## E1：异常发生率与 OCC 覆盖边界

回答 RQ1：

- 不同异常类型；
- 不同 Agent 数量；
- 不同 schedule；
- 不同模型。

结果应展示：

- 普通 OCC 能消除 A1；
- 对 A2–A6 仍存在明显漏检；
- structural correctness 不等于 semantic correctness。

## E2：依赖发现准确性

回答 RQ2：

比较：

- explicit dependency；
- trace-only；
- schema-only；
- semantic-only；
- hybrid。

## E3：冲突检测效果

比较所有 coordination 和 verifier baseline：

- precision；
- recall；
- F1；
- invalid commit；
- final state correctness。

## E4：端到端任务成功率

证明检测不是目的，最终任务恢复才是主要价值。

## E5：Targeted Repair 成本

比较：

- no repair；
- full rerun；
- downstream closure rerun；
- SemState minimum repair。

## E6：Provenance 价值

比较：

- version only；
- version + dependency；
- version + dependency + provenance。

观察：

- 根因定位；
- evidence mismatch；
- repair accuracy；
- 审计重建。

## E7：跨模型泛化

至少包含：

- 强闭源模型；
- 中等闭源/开源模型；
- 小型开源模型。

关键不是比较模型排行榜，而是证明：

> SemState 的相对收益不依赖单一模型。

---

# 13. 消融与鲁棒性实验

## 13.1 组件消融

1. Remove version validation；
2. Remove trace-derived dependencies；
3. Remove semantic dependency inference；
4. Remove provenance；
5. Remove deterministic validators；
6. Remove invalidation；
7. Replace targeted repair with full rerun。

## 13.2 依赖噪声

- 删除 10%、20%、30%、40% 的真实依赖；
- 添加 10%、20%、30% 的错误依赖；
- Agent 漏报 read-set；
- tool trace 缺失；
- source refs 缺失。

## 13.3 约束噪声

- 不完整 schema；
- 错误规则；
- 自然语言约束歧义；
- LLM verifier 判断错误。

## 13.4 扩展性

- 状态节点数量；
- 图深度；
- Agent 数量；
- 并发任务数量；
- repair subgraph 大小。

---

# 14. 错误分析

正文至少展示四类失败：

1. 隐含依赖未被发现；
2. LLM verifier 误判语义兼容；
3. 依赖图过度传播导致多余重跑；
4. 修复后的 Agent 重新引入旧假设。

每类失败应报告：

- 发生原因；
- 影响；
- 可否由 deterministic validator 修正；
- 是否属于方法边界；
- 后续改进方向。

---

# 15. AAAI 与 IJCAI 的差异化包装

## 15.1 AAAI 版本

重点：

- 新的多 Agent 失败问题；
- 自动依赖发现；
- 状态冲突检测；
- 端到端任务成功；
- 计划修复；
- Benchmark。

建议关键词方向：

- Multiagent Systems；
- Coordination and Collaboration；
- Agent Architectures；
- Distributed Problem Solving；
- Replanning and Plan Repair；
- Large Language Models。

写作原则：

- 第一页必须给出直观失败案例；
- 第二页结束前必须明确三项贡献；
- 主结果必须是最终任务成功率，不只是检测 F1；
- 工程细节压缩到附录；
- 不将 Event Store、OCC 本身作为主要创新。

## 15.2 IJCAI 版本

在 AAAI 版本基础上加强：

- 状态一致性的形式化定义；
- constraint representation；
- dependency reasoning；
- repair planning；
- 更大规模 Benchmark；
- 更强人工评价；
- 更多 Agent 和模型；
- 与知识表示、规划和多 Agent 协调工作的联系。

IJCAI 论文应更清楚解释：

- 语义状态有效性；
- 修复操作空间；
- 最小修复目标；
- 算法性质和复杂度；
- 不同依赖完整度下的行为。

---

# 16. 论文结构建议

由于 AAAI / IJCAI 正文空间紧张，建议按以下结构组织。

## 1. Introduction

- 一个跨键语义冲突案例；
- 传统版本检查为什么失败；
- SemState 方法概览；
- 三项贡献；
- 关键结果。

## 2. Related Work

仅保留：

- LLM multi-agent coordination；
- agent memory/shared state；
- workflow validation and repair；
- provenance and state consistency。

不要展开：

- 全部 RAG；
- 长上下文模型；
- 完整数据库历史；
- Agent OS 类比。

## 3. Problem Formulation

- versioned shared state graph；
- transaction envelope；
- semantic conflict；
- state validity；
- repair objective。

## 4. SemState

- dependency discovery；
- hybrid validation；
- invalidation；
- targeted repair。

## 5. SemStateBench

- domains；
- schedules；
- anomalies；
- ground truth；
- statistics。

## 6. Experiments

- E1 OCC limitation；
- E2 detection；
- E3 task success；
- E4 repair cost；
- ablations；
- error analysis。

## 7. Conclusion and Limitations

---

# 17. 原有 ARD 内容的取舍

## 17.1 主论文继续保留

- Event Store；
- Versioned State；
- TransactionManager；
- OCC；
- read-set；
- TraceStore；
- provenance；
- Planner/Executor/Verifier；
- 状态历史重建。

## 17.2 降为基础设施描述

- ContextPack；
- Context MMU；
- Hybrid Retrieval；
- Knowledge Store；
- Session/Working/Long-term 分层状态。

## 17.3 从主实验移除

- 48 条 QA 检索实验；
- BM25、vector、HyDE、RAPTOR 对比；
- 简洁性、完整性等普通问答评分；
- 8K vs 32K 上下文主张；
- “状态可以替代长上下文”的强结论；
- 通用 Agent OS 完整架构；
- 完整 MVCC 引擎表述。

## 17.4 单独输出

ARD 原论文可整理为：

- Workshop；
- Student Track；
- Demo；
- Technical Report；
- SemState 的系统技术报告。

---

# 18. AAAI-27 时间计划

## 18.1 官方节点

截至 2026-06-09，AAAI-27 主会时间节点为：

- 2026-06-17：OpenReview 作者注册开放；
- 2026-06-24：论文提交开放；
- 2026-07-21：摘要截止；
- 2026-07-28：全文截止；
- 2026-07-31：补充材料和代码截止。

## 18.2 七周冲刺计划

### 阶段 0：方向冻结  
**2026-06-09 至 2026-06-13**

任务：

- 冻结 SemState 主张；
- 删除 Agent OS 扩张计划；
- 定义 5–6 种异常；
- 确定四个任务领域；
- 确定状态 schema；
- 确定主结果指标。

交付：

- 2 页 problem formulation；
- anomaly taxonomy；
- benchmark schema；
- 系统改造任务单。

---

### 阶段 1：最小闭环实现  
**2026-06-14 至 2026-06-23**

任务：

- Typed State Graph；
- Transaction Envelope；
- trace-derived dependency；
- version/dependency validator；
- invalidation propagation；
- targeted repair MVP；
- 10–20 个可执行场景。

交付：

- SemState MVP；
- E1 小规模结果；
- 端到端 repair demo。

### 决策门槛 G1：2026-06-23

必须满足：

- 至少 4 类异常可稳定复现；
- read-set OCC 明显漏检跨键异常；
- SemState 能完成至少一种局部修复；
- 结果不是完全依赖人工 LLM Judge。

未满足：停止 AAAI 强行投稿，转入 IJCAI 完整路线。

---

### 阶段 2：Benchmark 与基线  
**2026-06-24 至 2026-07-05**

任务：

- 扩展至 40–60 个基础任务；
- 形成 250–400 个执行历史；
- 实现主要基线；
- 完成 3 个模型的首轮实验；
- 加入 deterministic validators。

交付：

- E1、E2、E3 主表初版；
- Benchmark 数据生成器；
- 基线复现实验。

### 决策门槛 G2：2026-07-05

必须满足：

- SemState 对 semantic conflict recall 有稳定提升；
- final state correctness 明显提升；
- targeted repair 成本低于 full rerun；
- 至少两个领域有效；
- 没有明显数据泄漏或评价器偏置。

---

### 阶段 3：完整实验  
**2026-07-06 至 2026-07-15**

任务：

- 扩展到 60–100 个任务；
- 500–800 个执行历史；
- 完成 3 个模型系列；
- 完成消融；
- 完成 dependency noise；
- 完成成本分析；
- 人工审核关键子集。

交付：

- 所有主表；
- 所有核心图；
- 错误分析；
- 附录素材。

### 决策门槛 G3：2026-07-15

必须满足：

- 主结论在随机重复后稳定；
- 至少一个关键指标显著提升；
- 方法优势不只来自 oracle dependency；
- 最终任务成功率提升；
- 论文已有完整初稿。

---

### 阶段 4：论文压缩与核验  
**2026-07-16 至 2026-07-28**

任务：

- 压缩正文；
- 重画方法图；
- 完成相关工作；
- 统一统计口径；
- 检查匿名性；
- 复现实验；
- 整理代码；
- 完成摘要和投稿信息。

禁止：

- 新增大模块；
- 临时更换主任务；
- 在最后一周重构系统；
- 只为提高数字而改变 Ground Truth。

---

### 阶段 5：补充材料  
**2026-07-29 至 2026-07-31**

内容：

- 详细算法；
- 完整 Benchmark；
- prompts；
- validators；
- 模型参数；
- 更多结果；
- 代码说明；
- 数据卡和伦理说明。

---

# 19. IJCAI 增强计划

截至 2026-06-09，IJCAI-27 的正式投稿日程尚未公布。IJCAI 2026 的主会采用 7 页正文加 2 页参考文献，并设两阶段评审；后续版本应以当届正式 CFP 为准。

若 AAAI-27 版本未投稿、被拒或需要增强，按以下路线继续：

## 阶段 I：扩大 Benchmark

- 150–250 个基础任务；
- 1,500–3,000 个执行历史；
- 增加数据流水线和基础设施领域；
- 增加真实项目实例；
- 公开 benchmark generator。

## 阶段 II：加强形式化

- semantic validity 的严格定义；
- dependency completeness assumption；
- validator soundness 条件；
- repair objective；
- 算法复杂度；
- 不完整依赖下的理论边界。

## 阶段 III：加强 AI 方法

可选择增加一项：

- learned dependency inference；
- calibrated semantic conflict classifier；
- repair policy learning；
- uncertainty-aware validation；
- adaptive validator routing。

## 阶段 IV：增强评价

- 4–6 个模型；
- 2–8 个 Agent；
- 人工双重标注；
- 更多真实工具调用；
- 真实代码、SQL 和配置验证；
- cross-domain transfer。

---

# 20. 项目管理与人员分工建议

## Track A：方法与系统

负责：

- Typed State Graph；
- Transaction Envelope；
- dependency collector；
- validators；
- invalidation；
- repair planner。

## Track B：Benchmark

负责：

- 场景模板；
- Ground Truth；
- validators；
- schedule generator；
- 数据质量审核。

## Track C：实验

负责：

- baselines；
- 模型运行；
- 成本统计；
- 显著性检验；
- 图表；
- 错误分析。

## Track D：论文

负责：

- problem formulation；
- related work；
- 方法描述；
- 结果叙事；
- 附录；
- 匿名与复现检查。

如果团队人数有限，应优先顺序为：

```text
Benchmark Ground Truth
> End-to-End Method
> Strong Baselines
> Final Task Success
> Ablation
> System UI / Engineering Polish
```

---

# 21. 拒稿风险与应对

## 风险 1：只是 OCC 的包装

应对：

- 核心案例必须是跨键、隐含依赖冲突；
- 主方法包含 semantic dependency inference；
- 报告 structural conflict 与 semantic conflict 的差异；
- 展示普通 OCC 通过但任务仍失败的案例。

## 风险 2：依赖图由人工提供

应对：

- oracle dependency 只作为性能上界；
- 主要结果使用 trace-derived 和 automatically inferred dependencies；
- 报告 dependency detection precision/recall；
- 做依赖噪声实验。

## 风险 3：Ground Truth 依赖 LLM Judge

应对：

- 软件、SQL、配置领域使用执行和 parser；
- 科研任务使用数值检查和人工标注；
- LLM Judge 仅负责无法规则化的开放语义；
- 报告人工与自动评分一致性。

## 风险 4：Benchmark 太人工

应对：

- 使用真实代码、schema、配置和研究材料；
- 公布场景生成规则；
- 包含自然出现的错误；
- 增加真实项目案例子集；
- 公开 task DAG、state graph 和 validators。

## 风险 5：只提高检测 F1，不提高任务成功

应对：

- 必须实现 repair；
- 主结果报告 end-to-end task success；
- 检测、阻止提交和恢复任务分开评价。

## 风险 6：修复方法等同于重跑全部任务

应对：

- 报告 minimal re-execution ratio；
- 与 full rerun 比较；
- 展示 unaffected branch 不被重跑；
- 统计 token、调用和延迟。

## 风险 7：论文内容过多

应对：

正文只保留：

- semantic conflict；
- dependency discovery；
- hybrid validation；
- targeted repair；
- SemStateBench。

其余进入附录或 ARD 技术报告。

## 风险 8：新颖性不足

应对：

- 持续更新 2025–2026 年相关工作；
- 建立“已有工作能力—SemState能力”精确矩阵；
- 不再声称首次引入事务、OCC、版本或 provenance；
- 把新颖性集中在自然语言共享状态的隐含依赖和增量修复。

---

# 22. 投稿前必须回答的审稿人问题

1. 为什么普通 read-set OCC 不能解决你们的问题？
2. Semantic conflict 与普通 contradiction detection 有何不同？
3. 依赖图从哪里来？
4. 如果依赖信息错误，系统是否仍然有效？
5. 为什么需要多 Agent，而不是单 Agent 多步骤任务？
6. Ground Truth 是否客观？
7. SemState 是否真的提高最终任务成功率？
8. 与全量串行和全量重跑相比，收益是什么？
9. 是否只对软件任务有效？
10. LLM verifier 是否引入新的不稳定性？
11. 方法是否依赖某个特定模型？
12. 修复后的状态是否可能再次冲突？
13. provenance 在方法中是必要组件还是附加功能？
14. 系统开销是否抵消并行收益？
15. Benchmark 是否存在模板泄漏或过拟合？

论文和实验必须在投稿前给出这些问题的明确答案。

---

# 23. 最低可投稿标准

## AAAI-27 最低 Go 条件

必须同时满足：

- 至少 4 个领域或 3 个强领域；
- 至少 60 个基础任务；
- 至少 500 个执行历史；
- 至少 3 个模型系列；
- 至少 5 个强基线；
- 客观 Ground Truth 占主要比例；
- semantic conflict F1 明显高于 OCC/verifier baseline；
- final task success 显著提高；
- repair cost 明显低于 full rerun；
- 非 oracle dependency 下仍有效；
- 完成匿名代码和补充材料。

## No-Go 条件

出现任一情况，应停止仓促冲刺，转投后续 IJCAI：

- 主要结果只在人工依赖图下成立；
- 只能展示少数案例；
- 全部评价依赖同一 LLM Judge；
- 检测提高但最终任务不提高；
- repair 等同于全量重跑；
- 结果只在一个模型或一个领域成立；
- 无法明确区别于 read-set OCC；
- 论文主线仍包含过多 Context MMU/RAG 内容。

---

# 24. 推荐论文主结果表

主表应尽量同时体现正确性和成本：

| Method | Conflict F1 ↑ | Invalid Commit ↓ | Final State Correctness ↑ | Task Success ↑ | Repair Calls ↓ | Token Cost ↓ |
|---|---:|---:|---:|---:|---:|---:|
| No Coordination | | | | | | |
| LWW | | | | | | |
| Global Lock | | | | | | |
| Read-set OCC | | | | | | |
| Verifier Agent | | | | | | |
| Full Rerun | | | | | | |
| SemState-Detect | | | | | | |
| SemState-Full | | | | | | |

理想主结论形式：

> Compared with read-set OCC, SemState reduces invalid commits by X%, improves end-to-end task success by Y points, and restores consistency using Z% of the model calls required by full workflow re-execution.

在获得真实实验结果前，不应预设或编造 X、Y、Z。

---

# 25. 推荐方法图

```text
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Workflow                     │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent Transaction Envelope                                 │
│ read_set | write_set | source_refs | tool_trace | task_id   │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ Dependency Discovery                                        │
│ Trace-derived | Structural | Latent Semantic                │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ Hybrid Validation                                           │
│ Version | Schema | Evidence | Execution | Semantic          │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
                  ┌────────────┴────────────┐
                  ↓                         ↓
             Valid Commit             Conflict Detected
                                            ↓
                              Dependency Invalidation
                                            ↓
                              Targeted Repair Planning
                                            ↓
                               Selective Re-execution
                                            ↓
                                Versioned State Commit
```

---

# 26. 推荐摘要逻辑模板

## 背景

LLM Agent 越来越多地通过共享状态协作完成长程任务，但现有协调机制主要检测同键版本冲突。

## 问题

多个 Agent 可能更新不同状态字段，却由于过期依赖或不兼容假设产生语义错误；这些提交可以通过普通 OCC。

## 方法

SemState 将共享状态建模为带版本、依赖和证据的状态图，通过 trace-derived dependency discovery、hybrid validation、dependency invalidation 和 targeted repair 管理语义冲突。

## Benchmark

构建 SemStateBench，在多个领域和并发调度中提供可执行约束与客观 Ground Truth。

## 结果

报告：

- 冲突检测；
- invalid commits；
- final task success；
- repair cost；
- 跨模型泛化。

## 结论

结构性版本正确不足以保证多 Agent 共享状态的语义正确；依赖感知验证和局部修复可以提高可靠性并减少全量重执行。

---

# 27. 最终执行结论

为了冲击 AAAI / IJCAI，团队应立即停止扩展完整 Agent OS，也不应继续把 Context MMU、长上下文对比和普通 Event Store 作为主论文创新。

当前最值得投入的路线是：

> **研究多 Agent 共享状态中“版本检查通过但语义仍然错误”的异常，提出自动依赖发现、混合语义验证、失效传播和最小子图修复，并以具有客观 Ground Truth 的多领域 Benchmark 证明其能够提高最终任务成功率。**

成功投稿所需的不是更多零散功能，而是：

1. 更明确的问题；
2. 更自动化的依赖发现；
3. 更客观的 Ground Truth；
4. 更强的端到端任务实验；
5. 更严格的基线与消融；
6. 更聚焦的七页论文叙事。

---

# 28. 一页行动清单

## 本周必须完成

- [ ] 冻结标题和主张；
- [ ] 定义 5–6 类异常；
- [ ] 定义 State Graph 和 Transaction Envelope；
- [ ] 确定四个领域；
- [ ] 设计 10 个端到端场景；
- [ ] 完成 read-set OCC 失败案例；
- [ ] 设计 targeted repair MVP。

## 两周内必须完成

- [ ] trace-derived dependency；
- [ ] deterministic validators；
- [ ] invalidation propagation；
- [ ] targeted repair；
- [ ] 20+ 可执行任务；
- [ ] 主要基线；
- [ ] 初步结果。

## 一个月内必须完成

- [ ] 60–100 个基础任务；
- [ ] 500–800 个执行历史；
- [ ] 3 个模型系列；
- [ ] 主实验；
- [ ] 消融；
- [ ] 成本分析；
- [ ] 人工核验；
- [ ] 完整论文初稿。

## 投稿前必须完成

- [ ] Go/No-Go 审核；
- [ ] 匿名检查；
- [ ] 统计复核；
- [ ] 代码复现；
- [ ] Benchmark 数据卡；
- [ ] 伦理与限制；
- [ ] 补充材料；
- [ ] 相关工作更新。
