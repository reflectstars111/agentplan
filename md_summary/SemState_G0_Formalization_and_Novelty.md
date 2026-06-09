# SemState G0：形式化定义、新颖性边界与实现状态

日期：2026-06-09

## 1. 研究对象

SemState 研究的是版本化共享产物的**提交时语义有效性**，不是一般的多
Agent 运行追踪，也不是失败发生后的根因定位。

设共享状态为：

\[
S = \{(k, v_k, x_k, \sigma_k)\}
\]

其中 \(k\) 是状态键，\(v_k\) 是单调版本，\(x_k\) 是值，
\(\sigma_k \in \{valid, stale, invalid, needs\_verification\}\)。

一个事务包定义为：

\[
T = (A, Q, R, W, D, E, \tau)
\]

- \(A\)：agent；
- \(Q\)：task；
- \(R=\{(k,v)\}\)：read set；
- \(W=\{(k,x')\}\)：write set；
- \(D=\{(s,t,v_s,o,c,h)\}\)：版本化依赖边；
- \(E=\{(s,v_s,claim)\}\)：证据引用；
- \(\tau\)：可观察执行 trace。

提交后的候选状态记为 \(S \oplus W\)。若 read-set OCC 通过，但
\(S \oplus W\) 违反依赖版本、schema、跨键领域约束、可执行检查或证据
版本约束，则称该提交为 **Semantic Invalid Commit**。

## 2. 五类核心异常

1. **Same-key conflict**：写入键自读取后版本变化。
2. **Cross-key stale dependency**：目标产物依赖的其他键已产生新版本。
3. **Derived artifact stale**：派生产物所记录的源版本落后于当前源版本。
4. **Cross-key constraint conflict**：每个单键写入分别合法，但组合状态违反
   部署容量、迁移兼容性或流水线 schema 等约束。
5. **Evidence-version mismatch**：提交使用的证据来自旧版本来源。

Partial Commit 是存储层故障测试，不作为同级语义异常。Missing
Invalidation 和 Invalid Repair 是方法失败结果。

## 3. 验证顺序

固定顺序为：

1. read-set 版本；
2. dependency 版本；
3. write schema；
4. 确定性领域规则；
5. 可执行检查；
6. evidence 版本；
7. 后续研究阶段才允许加入语义模型。

硬依赖不匹配必须拒绝；软依赖不匹配可提交为
`needs_verification`，但不能标记为 `valid`。

## 4. 修复定义

对冲突涉及节点集合 \(C\)，沿依赖图求下游闭包 \(Down(C)\)。修复候选为：

\[
Repair(C) = Down(C) - Revalidated(C)
\]

系统按拓扑顺序重跑其 producer task。该算法只主张选择受影响分支，不主张
求解全局最优重执行子图。

## 5. 与近期工作的边界

| 工作 | 主要问题 | 与 SemState 的重叠 | SemState 必须保持的区别 |
|---|---|---|---|
| Token Coherence, arXiv:2603.15183 | 用 MESI 类协议降低共享产物同步成本，并维护版本与失效状态 | 版本化产物、失效传播 | SemState 的主问题是提交前语义有效性与跨键约束，不以 token 节省或广播成本为主贡献 |
| GraphTracer, arXiv:2510.10581 | 用信息依赖图做多轮 Agent 失败归因与根因追踪 | 信息依赖图、错误传播 | SemState 在错误状态提交前执行确定性拒绝，并定义事务语义与最终状态正确性；不以事后归因为主任务 |
| MAST, arXiv:2503.13657 | 从执行 trace 建立多 Agent 失败分类，并提供 Judge 管线 | 失败分类、验证失败 | SemState 聚焦共享状态提交协议、可重建 Ground Truth 和局部修复，不提出通用失败 taxonomy |

重要引用修正：`arXiv:2503.13657` 是 MAST（Why Do Multi-Agent LLM
Systems Fail?），不是“ALAS”。在确认 ALAS 的正确书目信息前，不应在论文中
使用该名称或将其与该 arXiv ID 绑定。

来源：

- https://arxiv.org/abs/2603.15183
- https://arxiv.org/abs/2510.10581
- https://arxiv.org/abs/2503.13657

## 6. 当前实现

- `ard/infra/db.py`：可嵌套 SQLite 原子事务。
- `ard/store/transaction.py`：事件、投影、事务状态同一事务提交。
- `semstate/models.py`：固定研究接口。
- `semstate/validation.py`：有序确定性验证。
- `semstate/dependencies.py`：从 read set、任务输入、ContextPack、工具参数和
  source refs 收集可观察依赖。
- `semstate/runtime.py`：validate、commit、失效传播和 repair。
- `semstate/benchmark.py`：12 个手工 G0 案例。
- `semstate/histories.py`：40 个基础场景、每场景 6 个调度，共 240 条历史。
- `semstate/evaluation.py`：八类基线与主要指标。
- `semstate/statistics.py`：按基础场景配对的聚类 Bootstrap。
- `semstate/ground_truth.py`：与主验证器分离的 Ground Truth 重建。
- `semstate/noise.py`：缺失依赖和错误依赖的退化曲线。
- `semstate/experiments.py`：JSONL 结果与 append-only manifest 恢复。

## 7. 当前结果的正确解释

`eval_data/semstate_baselines_v1.json` 是确定性合成 sanity check：

- Read-set OCC Invalid Commit Rate：0.75；
- SemState Trace-only：0.25；
- SemState Full：0.00；
- SemState Full Final State Correctness：1.00。

这些数字只说明实现与构造案例一致，不能作为论文主结果。下一阶段必须加入：

- parser/schema/执行测试生成的独立 Ground Truth；
- 依赖缺失和错误依赖噪声曲线；
- DeepSeek 三次全量运行；
- 分层跨模型子集；
- token、延迟、repair call 与 task success 的真实测量。
