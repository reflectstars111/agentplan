# Agent Runtime Database（ARD）下一阶段研究计划

## 一、当前项目状态评估

### 已完成部分

目前已经完成：

* Agent Runtime整体架构设计
* Hierarchical State Architecture设计
* Context MMU设计
* Context Page Fault概念设计
* Hybrid Retrieval设计
* State Transaction设计
* MVCC状态演化设计
* Trace Store设计
* MVP路线图设计

已经形成较完整的系统蓝图。

---

### 当前最大问题

目前项目主要停留在：

```text
Architecture Design
```

阶段。

尚未证明：

1. Context MMU是否有效；
2. State Store是否必要；
3. Transaction机制是否带来收益；
4. ARD是否优于传统RAG系统。

因此下一阶段目标不应继续扩展架构，而应进入验证阶段。

---

# 二、重新聚焦研究问题

未来研究聚焦以下核心问题。

## RQ1

如何在固定Context Budget下提高知识利用效率？

对应模块：

* Context MMU
* Hybrid Retrieval

---

## RQ2

如何实现长期任务中的状态管理？

对应模块：

* State Store
* Knowledge Store

---

## RQ3

如何保证Agent状态更新的一致性与可追溯性？

对应模块：

* Transaction Manager
* MVCC
* Trace Store

---

# 三、最重要的研究假设

当前项目最核心假设：

> Context MMU能够比传统RAG构建更有效的上下文包，从而提升长期任务表现。

如果该假设不成立：

* Transaction没有意义；
* State Store没有意义；
* Agent Runtime Database失去基础。

因此必须优先验证。

---

# 四、未来两周任务

## Week 1

目标：

完成最小可验证原型设计。

### Task 1

设计统一State Schema。

定义：

* UserState
* ProjectState
* TaskState
* KnowledgeState

输出：

```text
state_schema.md
```

---

### Task 2

设计Context MMU流程。

明确：

* 输入
* 检索
* 排序
* 去重
* 压缩
* Budget分配

输出：

```text
context_mmu_design.md
```

---

### Task 3

准备测试数据。

推荐：

H2LLM相关材料：

* H2LLM论文
* PIM Simulator代码
* 自己的学习笔记
* 汇报材料

构建统一知识库。

---

## Week 2

目标：

完成第一版实验系统。

### Task 1

实现：

```text
Hybrid Retriever
```

包含：

* Vector Retrieval
* BM25 Retrieval
* Reranking

---

### Task 2

实现：

```text
Context MMU
```

包含：

* Candidate Merge
* Ranking
* Compression
* Context Packaging

---

### Task 3

实现：

```text
Trace Logger
```

记录：

* Query
* Retrieval
* Context
* Answer

---

# 五、实验设计

## Baseline 1

普通RAG

流程：

```text
Query
↓
Vector Search
↓
Top-K
↓
LLM
```

---

## Baseline 2

Hybrid RAG

流程：

```text
Vector
+
BM25
↓
Rerank
↓
LLM
```

---

## Proposed Method

ARD Context MMU

流程：

```text
Query
↓
Hybrid Retrieval
↓
Context MMU
↓
Budget Allocation
↓
Context Package
↓
LLM
```

---

# 六、评估指标

## Retrieval

* Precision@K
* Recall@K
* MRR

---

## Context

* Context Utilization
* Token Efficiency
* Evidence Coverage

---

## Task

* Question Accuracy
* Long-Horizon Consistency
* Citation Accuracy

---

## Runtime

* Latency
* Token Cost

---

# 七、数据库方向补课计划

当前Agent知识已经较多。

未来重点补数据库基础。

## 第一阶段

学习：

* ACID
* Transaction
* MVCC
* WAL
* Storage Engine

目标：

理解状态一致性。

---

## 第二阶段

学习：

* Query Planner
* Query Optimizer
* Index Structure

目标：

理解Context MMU未来演化方向。

---

## 第三阶段

阅读系统案例：

* PostgreSQL
* CockroachDB
* TiDB

目标：

理解大型状态管理系统设计。

---

# 八、未来三个月路线图

## Month 1

目标：

证明Context MMU有效。

成果：

* Prototype V1
* 第一版实验结果

---

## Month 2

目标：

加入State Store。

成果：

* State Runtime V1

验证：

长期任务连续性。

---

## Month 3

目标：

加入Transaction与MVCC。

成果：

* Agent Runtime Database Alpha

验证：

状态演化与一致性管理。

---

# 九、明确暂缓内容

未来三个月暂不做：

* 多Agent并发
* RL调度器
* Knowledge Graph全实现
* GUI系统
* 权限沙箱
* 自动长期运行

原因：

这些无法验证核心研究假设。

---

# 十、项目当前最重要目标

当前最重要目标不是：

```text
继续设计架构
```

而是：

```text
证明Context MMU成立
```

如果Context MMU能够显著提升：

* 长任务连续性
* 上下文利用率
* Token效率

则ARD具有继续发展的价值。

否则应重新审视整体架构设计。

---

# 一句话总结

未来阶段的核心任务：

从“设计Agent Runtime Database”，转向“证明Context MMU能够成为Agent系统中的虚拟内存管理器”。
