# SemState AAAI-27 修订执行计划

## 总体判断

- 从 ARD 转向 SemState 的方向合理：问题更集中，也更符合多 Agent 可靠性研究。
- 原计划按当前范围执行的可行性较低。单人七周同时完成四领域 Benchmark、自动语义依赖、混合验证、最小修复、三模型全量实验，工作量过大。
- AAAI-27 官方节点确认为摘要 **2026-07-21**、全文 **2026-07-28**、补充材料 **2026-07-31**，见 [官方页面](https://aaai.org/conference/aaai/aaai-27/)。
- 采用质量优先策略：设置硬性 Go/No-Go 门槛，证据不足则转 IJCAI。

当前基础不能直接作为可靠实验底座：

- [TransactionManager](/F:/agentplan/ard/store/transaction.py:92) 的多事件提交并非原子操作，部分写入可能残留。
- `src/` 与 `ard/` 是两套基本独立的运行时，不应在七周内全面合并。
- 当前依赖图仅支持代码 import 关系，现有 Verifier 主要是关键词启发式。
- 旧 8K/32K 实验实际仅使用 426–594 tokens，见 [experiment_result4.txt](/F:/agentplan/aru_result/experiment_result4.txt:20)，不能作为上下文容量证据。

## 研究收缩

- 将核心问题改为：**Semantic Invalid Commit**，即多个更新分别通过版本检查，但组合状态违反跨键依赖或领域约束。
- 可将其中典型情况定义为“semantic write skew”，但不声称所有语义冲突都属于数据库 write skew。
- 论文仅保留三项贡献：
  1. OCC 可通过但语义错误的异常定义与 SemStateBench。
  2. 依赖感知的提交时混合验证。
  3. 状态失效传播与选择性重执行。
- 将五个 RQ 合并为三个：OCC 覆盖边界、检测与最终状态正确性、修复效果与成本。
- AAAI 正文异常类型限定为：同键冲突、跨键过期依赖、派生产物过期、跨键约束冲突、证据版本错配。
- Partial Commit 作为存储正确性故障测试；Missing Invalidation 和 Invalid Repair 作为方法失败结果，不再作为同级输入异常。
- 自动依赖发现以 observable trace 和结构化 schema 为主；latent semantic dependency 仅作增强项，不能成为主结果成立的前提。
- Provenance 保留为消融；Context MMU、RAG 和上下文长度实验移出主论文。

新颖性需要在 6 月 12 日前完成硬审核。近期工作已覆盖共享状态一致性、依赖图和局部修复，例如 [Token Coherence](https://arxiv.org/abs/2603.15183)、[GraphTracer](https://arxiv.org/abs/2510.10581) 和 [ALAS](https://arxiv.org/abs/2503.13657)。SemState 必须明确区别为：**面向版本化共享产物的提交时语义有效性，而非一般运行时追踪或失败后定位。**

## 实现方案

建立独立 `semstate/` 研究包，不接入 Web、GUI 和通用 Agent-OS 产品流程。

核心接口固定为：

- `TransactionEnvelope`：agent、task、read/write set、依赖、证据和 trace。
- `StateNode`：key、version、type、status、producer task、source refs。
- `DependencyEdge`：source、target、origin、confidence、hard/soft。
- `ValidationDecision`：`commit | reject | mark_uncertain`、异常类型、证据和受影响状态。
- `RepairPlan`：失效节点、需要重跑的任务、拓扑顺序和估计成本。
- `SemStateRuntime.validate(envelope)`
- `SemStateRuntime.commit(envelope)`
- `SemStateRuntime.repair(conflict_id)`

实施顺序：

1. 修复 ARD 事务原子性，使事件、事务状态和投影在同一 SQLite 事务中提交；失败后不得残留事件或投影。
2. 建立版本化状态节点和依赖边投影。
3. 从 read set、任务输入、ContextPack、工具参数和 source refs 收集确定性依赖。
4. 按版本、依赖版本、schema、领域规则、可执行测试、证据版本、语义模型的顺序验证。
5. hard edge 变化传播为 `stale/invalid`；soft edge 传播为 `needs_verification`。
6. 修复算法采用“受影响下游闭包减去重新验证通过节点”，不声称求解全局最优子图。
7. 建立可恢复、可并行、保存原始 JSONL 的实验运行器。

## Benchmark 与实验

AAAI 版本限定三个可客观验证领域：

- 软件部署与配置；
- 数据库迁移；
- 数据处理流水线。

规模固定为：

- 40 个基础场景；
- 每场景 6 种调度，共 240 个规范执行历史；
- DeepSeek 全量运行 3 次，共 720 次；
- 分层抽取 80 个历史，在一个 OpenAI 和一个 Anthropic 模型上各重复 2 次，共 320 次；
- 总量约 1,040 次，而不是原计划隐含的 4,500–7,200 次；
- 至少 80% Ground Truth 来自 parser、schema、执行测试或确定性规则。

主基线限定为：No Validation、LWW、Read-set OCC、Oracle Dependency、Independent Verifier、Full Rerun、SemState Trace-only、SemState Full。

主要指标：

- Primary：Invalid Commit Rate、Final State Correctness。
- Secondary：Conflict F1、False Rejection、Task Success、Repair Calls、Tokens、Latency。
- 使用相同历史的配对比较，并按基础场景进行聚类 Bootstrap，避免把重复运行当作独立样本。

## 时间与门槛

- **6 月 9–12 日，G0**：完成新颖性矩阵、形式化定义、五类异常和 12 个手工验证案例。
- **6 月 13–20 日，G1**：修复事务层；完成两个领域、四类语义异常、OCC 漏检案例和一次局部修复闭环。
- **6 月 21 日–7 月 2 日，G2**：完成 120 个历史、主要基线和 DeepSeek 首轮结果；论文方法与 Benchmark 章节同步成稿。
- **7 月 3–13 日，G3**：完成 240 个历史、消融、依赖噪声和成本实验。
- **7 月 14–20 日**：完成跨模型子集、统计复核和全文；7 月 21 日提交摘要。
- **7 月 21–28 日**：仅复现、压缩、匿名检查和修订，不新增方法。
- **7 月 29–31 日**：补充材料、数据卡和匿名代码。

最终 Go 条件：

- SemState 对 Read-set OCC 的 Invalid Commit Rate 和 Final State Correctness 改善，其 95% 配对置信区间不跨零。
- False Rejection Rate 不高于 10%。
- 修复成功率与 Full Rerun 相差不超过 5 个百分点，同时模型调用量至少降低 30%。
- 主结果不依赖 oracle dependency 或 LLM Judge。
- 至少三个领域、两个额外模型上的变化方向一致。

任一核心条件未满足，则停止 AAAI 冲刺，保留完整结果转向 IJCAI。

## 验证要求

必须覆盖：

- 批量提交失败后事件和投影均不残留；
- Read-set OCC 可通过但跨键状态无效；
- deterministic validator 能拒绝错误提交；
- hard/soft dependency 的失效传播不同；
- 修复只重跑受影响分支；
- evidence-version mismatch 可定位到旧来源；
- 缺失和错误依赖下的退化曲线；
- Benchmark Ground Truth 可由独立验证器重建；
- 中断后的实验可以从 manifest 安全恢复。

默认假设：单人全职、独立研究包、DeepSeek 负责全量实验、其他模型仅运行分层子集，不进行 UI 或产品化集成。
