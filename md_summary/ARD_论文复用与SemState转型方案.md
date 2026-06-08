# ARD 现有论文复用与新研究方向转型方案

## 0. 文档目的

本文档用于回答两个核心问题：

1. **现有 ARD 论文是否仍有价值，哪些部分可以继续复用？**
2. **如何在现有系统和实验基础上，转向一个更清晰、更可辩护的新论文方向？**

本文给出的结论不是“放弃现有论文”，而是：

> 将 ARD 从“最终论文贡献本体”调整为“研究多 Agent 共享状态语义一致性的实验运行时与技术底座”。

现有工作中的 Event Store、版本状态、OCC、TraceStore、provenance 机制和初步实验均具有复用价值。真正需要大幅调整的是论文定位、研究问题、贡献声明和核心实验。

---

# 1. 核心判断

## 1.1 现有论文仍然有用

现有 ARD 论文已经完成了以下重要资产：

- provenance-aware ContextPack；
- Context MMU；
- append-only Event Store；
- versioned StateStore；
- TransactionManager；
- optimistic concurrency control；
- point-in-time state query；
- TraceStore；
- provenance、persistent state、context capacity 三类初步实验；
- OCC 与版本恢复的功能验证。

这些内容已经构成一个较完整的实验平台，不需要推倒重来。

## 1.2 当前问题不在实现，而在论文主张

现有论文试图同时证明：

- 一个新的通用 Agent Runtime；
- provenance-aware context assembly 的有效性；
- persistent state 的普遍价值；
- 小上下文可以被状态替代；
- event sourcing、OCC 和版本状态适用于 Agent 系统。

这一叙事范围过大，而且不同部分的实验证据强度不一致。

当前最强证据是 provenance 实验；状态实验仍属于探索性；上下文补偿实验样本较小；OCC 和版本机制主要是功能验证。

因此，现有论文不适合继续以“完整通用 Runtime 的整体创新”为唯一主线。

---

# 2. 推荐的新研究方向

## 2.1 方向名称

推荐将后续主论文转向：

> **面向多 Agent 共享状态的语义可串行化与依赖感知修复**

英文表述：

> **Semantic Serializability and Dependency-Aware Repair for Shared State in Multi-Agent LLM Systems**

推荐论文标题：

### 方案一

**SemState: Semantic Serializability for Shared State in Multi-Agent LLM Systems**

### 方案二

**Beyond OCC: Dependency-Aware State Coordination for Multi-Agent LLM Workflows**

### 方案三

**When Version Checks Are Not Enough: Semantic Race Conditions in Multi-Agent LLM Systems**

其中第三个标题最能直接体现问题意识。

---

# 3. 为什么这个方向比当前主线更明确

传统 OCC 可以识别：

- 两个 Agent 同时修改同一个 key；
- Agent 基于旧版本读取后提交；
- 同一状态发生 lost update。

但是，多个 Agent 即使修改的是不同 key，也可能产生语义冲突。

例如：

```text
Agent A:
database_engine = PostgreSQL

Agent B:
migration_script = 使用 MySQL 方言生成
```

这两个写入在数据库层面没有 write-write conflict，因为它们修改的是不同字段。

但在项目语义上，最终状态已经不一致。

因此，需要研究的问题不是普通的键级冲突，而是：

> 多 Agent 对共享项目状态的并发修改，是否能够等价于某个合法、语义一致的串行执行？

这就是 **Semantic Serializability**。

---

# 4. 新方向的核心研究对象

## 4.1 Typed Shared State Graph

将共享状态建模为带类型和依赖关系的状态图。

示例：

```text
ProjectState
├── architecture
├── database_engine
├── data_schema
├── migration_plan
├── api_contract
├── test_plan
└── deployment_plan
```

依赖关系示例：

```text
database_engine
→ data_schema
→ migration_plan
→ test_plan
→ deployment_plan
```

当 `database_engine` 发生变化时，系统不能只更新这一字段，还应判断：

- `data_schema` 是否仍然有效；
- `migration_plan` 是否需要重新生成；
- `test_plan` 是否需要重新验证；
- `deployment_plan` 是否受影响。

## 4.2 状态节点定义

建议状态节点至少包含：

```json
{
  "key": "migration_plan",
  "value": "...",
  "type": "derived_artifact",
  "version": 4,
  "status": "active",
  "read_dependencies": [
    {
      "key": "database_engine",
      "version": 2
    },
    {
      "key": "data_schema",
      "version": 5
    }
  ],
  "source_refs": [
    "decision_023",
    "schema_doc_v3"
  ],
  "trust_level": "verified_internal",
  "created_by": "agent_migration",
  "derived_from": [
    "database_engine@v2",
    "data_schema@v5"
  ],
  "supersedes": "migration_plan@v3"
}
```

## 4.3 Agent Transaction Envelope

每个 Agent 提交状态更新时，应附带：

```json
{
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
  ]
}
```

这样系统不仅知道 Agent 写了什么，还知道：

- 它读取了哪些状态；
- 它基于哪些版本推理；
- 它依赖哪些状态；
- 它使用了哪些证据；
- 哪些下游状态可能被本次提交影响。

---

# 5. 需要定义的语义异常类型

建议将以下异常定义为 benchmark 和论文中的正式 taxonomy。

## A1. Same-Key Lost Update

两个 Agent 同时修改同一个状态字段，后写覆盖前写。

## A2. Cross-Key Stale Read

Agent A 修改了上游状态；Agent B 仍基于旧上游状态更新另一个字段。

示例：

```text
A 更新 database_engine
B 仍基于旧 database_engine 生成 migration_plan
```

## A3. Derived Artifact Staleness

上游决策已经改变，但旧版本生成的派生产物仍被标记为 active。

## A4. Partial Semantic Commit

一个任务需要同时修改多个相关状态，但只有部分字段成功提交。

## A5. Evidence-Version Mismatch

状态内容已经更新，但 source reference 仍指向旧文档或旧决策版本。

## A6. Conflicting Agent Assumptions

两个 Agent 对同一项目形成不同隐含假设，但由于修改不同字段，普通 OCC 无法检测。

## A7. Dependency Propagation Failure

上游状态更新后，依赖它的下游状态没有被标记为 stale、invalid 或 needs_verification。

## A8. Invalid Repair

系统发现冲突后进行了不必要或错误的重执行，导致更多状态被覆盖。

---

# 6. 建议的研究问题

## RQ1：传统 OCC 无法检测哪些多 Agent 语义竞态？

目标：

- 建立异常分类；
- 测量不同调度下异常发生率；
- 区分 structural conflict 和 semantic conflict。

核心假设：

> Per-key OCC 可以防止 lost update，但无法充分防止跨 key 的语义不一致。

## RQ2：依赖感知验证能否提高共享状态一致性？

比较：

- No Coordination；
- Last Write Wins；
- Global Lock；
- Per-key OCC；
- Read-set OCC；
- Dependency-aware OCC；
- Semantic Constraint Validation。

核心观察：

- 结构冲突检测率；
- 语义冲突检测率；
- 最终状态正确性；
- 误报率；
- 系统成本。

## RQ3：依赖感知修复能否以最小代价恢复一致性？

研究检测到冲突后，系统如何处理：

1. 找到受影响状态；
2. 标记派生状态失效；
3. 计算最小重执行子图；
4. 重新读取最新版本；
5. 重新运行相关 Agent；
6. 创建新状态版本；
7. 保留原始事件和完整 provenance。

核心问题：

> 是否可以只重跑受影响的最小子图，而不是回滚或重跑整个工作流？

## RQ4：Provenance 是否能改善冲突判定与根因定位？

比较：

- Version only；
- Version + Dependency；
- Version + Dependency + Provenance；
- Full Semantic Repair。

观察 provenance 是否改善：

- 冲突来源判定；
- 状态可信度判断；
- 根因定位；
- 错误恢复；
- 审计重建。

---

# 7. 推荐的系统结构

后续论文不需要继续扩展完整 Agent OS，只保留以下模块：

```text
Typed Shared State Graph
        ↓
Transaction Envelope
        ↓
Structural OCC
        ↓
Dependency Validator
        ↓
Semantic Constraint Checker
        ↓
Dependency Invalidation
        ↓
Targeted Repair
        ↓
Event Store + TraceStore
```

## 7.1 Structural OCC

复用当前 TransactionManager，检查：

- read version 是否变化；
- 同一个 key 是否已被修改；
- 事务是否基于旧状态；
- 是否存在 lost update。

## 7.2 Dependency Validator

检查：

- Agent 是否声明了所有实际依赖；
- 上游版本是否仍有效；
- 当前写入是否依赖已过期状态；
- 派生状态是否引用正确版本。

## 7.3 Semantic Constraint Checker

优先采用：

- schema；
- typed constraints；
- deterministic rules；
- unit tests；
- domain validators。

仅在无法结构化验证的内容上使用 LLM Judge。

## 7.4 Dependency Invalidation

当上游状态发生变化时，将下游节点标记为：

- `stale`；
- `needs_verification`；
- `invalid`；
- `potentially_stale`。

## 7.5 Targeted Repair

根据依赖图计算需要重执行的最小任务集合。

例如：

```text
database_engine changed
→ migration_plan stale
→ test_plan needs_verification
→ deployment_plan potentially_stale
```

系统只重新执行这些相关任务，而不是重跑整个项目流程。

---

# 8. Benchmark 设计

## 8.1 Benchmark 名称

建议：

> **MASS-Consistency: Multi-Agent Shared State Consistency Benchmark**

或：

> **SemStateBench**

## 8.2 任务领域

建议先选择四类具有明确跨字段依赖的任务。

### 软件项目设计

状态包括：

- 架构；
- 数据库；
- API；
- 数据模型；
- 测试；
- 部署。

### 数据库迁移

状态包括：

- 源数据库；
- 目标数据库；
- schema 映射；
- 迁移脚本；
- 回滚策略；
- 验证结果。

### 研究报告协作

状态包括：

- 研究问题；
- 数据集；
- 实验设置；
- 分析结果；
- 结论；
- 引用来源。

### 事故响应

状态包括：

- 故障原因；
- 服务状态；
- 临时修复；
- 用户通知；
- 后续行动；
- 根因分析。

## 8.3 每个场景需要包含

- 初始状态；
- Agent 角色；
- Task DAG；
- 状态依赖图；
- 合法状态约束；
- 预设更新事件；
- 并发执行 schedule；
- 合法串行结果；
- 可验证的最终状态；
- 错误状态样例。

## 8.4 并发调度

对同一场景控制不同 schedule：

```text
Schedule 1:
A read → A write → B read → B write

Schedule 2:
A read → B read → A write → B write

Schedule 3:
A read old state → B update → A commit

Schedule 4:
A update parent state
B commit derived state based on old parent

Schedule 5:
A and B update different keys with incompatible assumptions
```

这样可以准确识别异常是由哪一种并发顺序触发的。

---

# 9. 实验对照组

建议比较以下方法：

| 方法 | 特点 |
|---|---|
| No Coordination | 所有 Agent 直接写共享状态 |
| Last Write Wins | 后写覆盖前写 |
| Global Lock | 全局串行化 |
| Per-key Lock | 按 key 加锁 |
| Basic OCC | 版本检查 |
| Read-set OCC | 检查实际读取版本 |
| Dependency-aware OCC | 检查跨 key 依赖 |
| Semantic Validation | 加入语义约束 |
| Full Repair | 检测、失效传播和最小重执行 |

---

# 10. 实验指标

## 10.1 一致性指标

- Structural Conflict Detection Precision；
- Structural Conflict Detection Recall；
- Semantic Conflict Detection Precision；
- Semantic Conflict Detection Recall；
- Semantic Serializability Rate；
- Invalid Commit Rate；
- Stale Derived Artifact Rate；
- Dependency Invalidation Accuracy。

## 10.2 任务指标

- Final State Correctness；
- Final Task Success；
- Constraint Satisfaction Rate；
- Cross-Agent Consistency；
- Evidence Consistency；
- Final Output Quality。

## 10.3 修复指标

- Repair Success Rate；
- Minimal Re-execution Ratio；
- Unnecessary Rollback Rate；
- Retry Count；
- Time to Consistency；
- Reintroduced Error Rate。

## 10.4 系统成本

- Token Cost；
- LLM Calls per Commit；
- Commit Latency；
- p50/p95/p99 Latency；
- Throughput；
- Conflict Retry Overhead。

## 10.5 可审计性

- State Lineage Completeness；
- Conflict Reconstruction Accuracy；
- Root-cause Localization Accuracy；
- Evidence-Version Alignment；
- Repair Trace Completeness。

---

# 11. 现有 ARD 资产的复用方式

## 11.1 可以直接保留的部分

| 现有资产 | 复用方式 |
|---|---|
| Event Store | 继续作为状态变更的 canonical log |
| StateStore | 增加 typed state 与 dependency 字段 |
| seq_num | 继续作为版本标识 |
| TransactionManager | 升级为 dependency-aware OCC |
| read_set | 扩展为 read_set + dependency_set |
| history() | 用于冲突重建和根因定位 |
| point-in-time query | 用于重建旧版本状态 |
| TraceStore | 增加 state lineage 和 repair trace |
| provenance | 用于证据版本对齐和冲突判断 |
| Shared Blackboard | 作为多 Agent 共享状态空间 |

## 11.2 需要增强的部分

| 现有模块 | 需要增加 |
|---|---|
| State schema | type、dependency、validity、status |
| Transaction | dependency_set、evidence_refs |
| Verifier | semantic constraint validation |
| Writeback | invalidation propagation |
| Scheduler | targeted repair scheduling |
| Trace | dependency and repair lineage |
| OCC | cross-key dependency validation |

## 11.3 应降级为辅助内容的部分

- 完整六步 Context MMU；
- 8K 与 32K 的主结论；
- 普通 persistent state 的整体收益；
- 类冯诺依曼 Agent OS 叙事；
- 普通 hybrid retrieval；
- 单线程 lost-update 测试。

---

# 12. 当前实验如何处理

## 12.1 Provenance 实验

保留，但不再作为整篇论文唯一核心贡献。

可用于证明：

- evidence reference 对冲突判断有帮助；
- provenance 对根因定位有帮助；
- source/version 对状态审计有帮助。

建议新增比较：

```text
Version only
vs
Version + Dependency
vs
Version + Dependency + Provenance
vs
Full Semantic Repair
```

## 12.2 Persistent State 实验

当前 n=10 的结果可保留为 pilot study。

作用：

- 说明状态收益具有任务依赖性；
- 支持“状态本身可能造成干扰”的动机；
- 为 benchmark 选择高状态依赖任务。

不要继续将其写成确认性结论。

## 12.3 Context Capacity 实验

保留在附录或 exploratory section。

表述改为：

> 初步结果显示，有限上下文下结构化状态可能更有价值，但需要更大规模实验验证。

## 12.4 OCC 与版本恢复实验

保留为 Structural OCC baseline。

后续增加：

- cross-key semantic conflict；
- derived artifact staleness；
- evidence-version mismatch；
- dependency propagation failure；
- targeted repair。

---

# 13. 论文结构调整

## 13.1 新论文结构

### 1. Introduction

提出：

- 多 Agent 并发修改共享状态；
- 普通版本检查只能发现结构冲突；
- 不同字段之间仍可能语义不一致；
- 需要语义可串行化和依赖感知修复。

### 2. Related Work

包括：

- Agent memory；
- persistent agent runtime；
- event-sourced state；
- OCC and concurrency control；
- multi-agent shared memory；
- semantic consistency；
- provenance and traceability。

### 3. Problem Formulation

正式定义：

- typed shared state；
- dependency graph；
- semantic race condition；
- semantic serializability；
- derived-state staleness；
- valid repair。

### 4. System Design

介绍：

- ARD runtime；
- transaction envelope；
- structural OCC；
- dependency validation；
- semantic constraints；
- invalidation；
- targeted repair。

### 5. Benchmark

介绍：

- 任务领域；
- 状态图；
- 并发 schedule；
- anomaly taxonomy；
- ground truth。

### 6. Experiments

依次回答：

- RQ1：异常类型与 OCC 局限；
- RQ2：依赖验证效果；
- RQ3：修复效果；
- RQ4：provenance 价值；
- 成本与延迟。

### 7. Discussion

讨论：

- 结构一致性与语义一致性的区别；
- 哪些约束可以确定性验证；
- LLM Judge 的边界；
- 修复成本；
- 泛化范围。

### 8. Limitations

说明：

- benchmark 规模；
- 状态 schema 人工定义；
- 部分语义约束依赖领域知识；
- 模型数量；
- 真实生产并发尚未验证。

### 9. Conclusion

总结：

- 传统 OCC 不足以保证语义一致；
- 依赖感知验证可识别跨 key 异常；
- targeted repair 可降低重执行成本；
- provenance 改善审计和根因定位。

---

# 14. 当前 ARD 论文的处理方案

## 14.1 路线 A：整理为 Workshop / Student Track 论文

保留当前 ARD 论文作为阶段性成果。

需要修改：

1. 修复条件数量与结果表不一致；
2. 补齐关键 provenance 对照；
3. 降低“首次”“普遍有效”等表述；
4. 将状态实验标记为 exploratory；
5. 将上下文实验标记为 pilot；
6. 将 OCC 与版本恢复标记为 functional validation；
7. 更新最新 Related Work；
8. 明确系统贡献与实验贡献的边界。

适合：

- Workshop；
- Student Track；
- Demo；
- Technical Report；
- 阶段性项目成果。

## 14.2 路线 B：作为新主论文的系统底座

将 ARD 定义为：

> An experimental runtime substrate for studying semantic consistency in versioned multi-agent state.

也就是说：

- ARD 是运行时平台；
- SemState 是新的研究机制；
- 旧实验是预实验和组件验证；
- 新论文核心是 semantic race、serializability 和 repair。

---

# 15. 推荐的双成果策略

## 成果一：ARD 技术报告或 Workshop 论文

主题：

> Provenance-aware context、persistent state 和 versioned runtime 的系统原型与初步实验。

目标：

- 保留现有工作；
- 形成可引用的阶段成果；
- 公开代码和实验框架；
- 不承担过强理论主张。

## 成果二：SemState 主论文

主题：

> 多 Agent 共享状态中的语义竞态、语义可串行化和依赖感知修复。

目标：

- 定义新问题；
- 构建 benchmark；
- 提出新机制；
- 完成严格对照；
- 形成更强的论文贡献。

---

# 16. 开发与实验路线图

## 阶段 1：重构状态模型

任务：

- 定义 typed state schema；
- 增加 dependency_set；
- 增加 evidence_refs；
- 增加 status 与 validity；
- 建立 State Dependency Graph。

产出：

- 新 StateItem schema；
- 新 Transaction Envelope；
- 依赖图存储模块。

## 阶段 2：构建异常检测

任务：

- 复用基本 OCC；
- 实现 cross-key dependency check；
- 实现 stale derived state 检测；
- 实现 evidence-version alignment；
- 实现 semantic constraint checker。

产出：

- anomaly detector；
- structural vs semantic conflict 分类结果。

## 阶段 3：实现失效传播与修复

任务：

- 计算受影响下游节点；
- 标记 stale/invalid；
- 计算最小重执行子图；
- 调度相关 Agent 重试；
- 写入新版本；
- 记录 repair trace。

产出：

- invalidation engine；
- targeted repair engine。

## 阶段 4：构建 Benchmark

最低建议：

- 30 个基础场景；
- 每个场景 4 种以上 schedule；
- 4 个任务领域；
- 3 个 Agent 角色；
- 至少 120 个执行案例；
- 明确的状态依赖和最终正确状态。

## 阶段 5：完成对照实验

比较：

- LWW；
- lock；
- per-key OCC；
- read-set OCC；
- dependency-aware validation；
- semantic repair。

报告：

- 一致性；
- 任务质量；
- 修复效果；
- 成本；
- 延迟；
- 审计能力。

---

# 17. 最低可行论文版本

如果资源有限，建议只实现：

```text
Typed State
+ Read-set OCC
+ Dependency Graph
+ Cross-key Validation
+ Invalidation
+ Targeted Repair
```

实验规模：

- 30 个场景；
- 4 类异常；
- 4 个 baseline；
- 1 个主模型；
- 1 个独立 Judge；
- 人工抽样验证。

不需要第一版就实现：

- 完整分布式数据库；
- 大规模向量检索；
- 复杂 GUI；
- 完整权限系统；
- 完整 Agent OS；
- 强化学习调度；
- 多租户系统。

---

# 18. 风险与应对

## 风险 1：语义一致性过于主观

应对：

- 使用 typed state；
- 定义 dependency DAG；
- 使用 deterministic constraints；
- 使用单元测试和 schema；
- 只对无法规则化的内容使用 LLM Judge。

## 风险 2：与普通 OCC 工作过于接近

应对：

明确区分：

- 普通 OCC：同 key 和版本冲突；
- 本研究：cross-key dependency、derived-state staleness、semantic serializability。

## 风险 3：系统范围再次膨胀

应对：

主论文只做：

- 状态图；
- OCC；
- 依赖验证；
- 失效传播；
- targeted repair。

Context MMU、多级记忆、权限系统和 Agent OS 放入背景或未来工作。

## 风险 4：Benchmark 过于人工

应对：

- 使用真实软件、数据库迁移、研究协作任务模板；
- 公开场景生成规则；
- 提供 deterministic ground truth；
- 报告人工和自动评价一致性。

---

# 19. 最终论文主张

后续论文不应再主要声称：

> 我们提出了一个新的通用 Agent Runtime。

推荐改为：

> 多 Agent 系统中的共享状态即使通过传统版本检查，也可能因为跨字段依赖和派生状态失效而产生语义竞态。我们将共享状态建模为带版本、依赖和证据的状态图，定义语义可串行化，并提出依赖感知验证、失效传播和最小范围重执行机制。

对应英文主张：

> Traditional version checks prevent same-key races but do not guarantee semantic consistency across interdependent state fields. We model multi-agent shared state as a versioned dependency graph, define semantic serializability, and introduce dependency-aware validation, invalidation, and targeted repair.

---

# 20. 最终结论

现有 ARD 论文没有失效。

其最有价值的部分包括：

- Event Store；
- Versioned State；
- TransactionManager；
- OCC；
- TraceStore；
- provenance；
- 初步状态实验。

但 ARD 更适合从“最终研究结论”转变为“新研究的实验基础设施”。

推荐形成以下关系：

```text
ARD Runtime
    ↓
Versioned Shared State
    ↓
Structural OCC
    ↓
SemState
    ↓
Semantic Serializability
    ↓
Dependency-Aware Repair
```

最终研究主线应是：

> **传统版本检查能够发现 Agent 是否修改了相同状态，却无法判断它们对不同状态字段的修改是否在语义上兼容。通过版本化依赖图、语义约束和最小范围修复，可以提高多 Agent 长程任务中的共享状态一致性，同时保留可追踪、可恢复和可审计的执行历史。**

这条路线能够最大程度复用现有代码与实验，同时形成一个比“通用 Agent Runtime”更加明确、可形式化、可实验和可辩护的论文方向。
