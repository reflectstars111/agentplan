# Agent-OS

面向多智能体系统的类冯诺依曼运行时架构 — Python + SQLite + FAISS + React。

> 借鉴计算机体系结构与操作系统的核心思想，通过分层记忆、上下文虚拟化、多级索引、任务调度和结果验证机制，使 Agent 能够在有限上下文窗口内稳定调用长期历史与大规模外部资料，完成复杂任务的可追踪执行。

## 系统架构

```
┌────────────────────────────────────────────────────────┐
│                   Agent-OS 运行时                       │
│                                                         │
│  POST /query ─→ AgentRuntime（简单问答模式）             │
│  POST /task  ─→ Controller → Scheduler（任务图模式）     │
│                                                         │
│  ┌─ 控制层 ─────────────────────────────────────┐      │
│  │ IntentDecoder → Planner → TaskGraph(DAG)      │      │
│  │ Scheduler（拓扑执行 + 并行）                    │      │
│  │ AgentRegistry + SharedBlackboard              │      │
│  └───────────────────────────────────────────────┘      │
│                                                         │
│  ┌─ 索引层 ─────────────────────────────────────┐      │
│  │ HybridRetriever（8 因子加权评分）              │      │
│  │   ├─ VectorIndex  （FAISS + BGE-M3 1024维）  │      │
│  │   ├─ KeywordIndex （SQLite FTS5 BM25）        │      │
│  │   └─ StructureIndex（层次结构树）              │      │
│  │ CodeParser（tree-sitter: Python/JS/TS）       │      │
│  │ PDFParser（OpenDataLoader + PyMuPDF 回退）     │      │
│  └───────────────────────────────────────────────┘      │
│                                                         │
│  ┌─ 上下文层 ───────────────────────────────────┐      │
│  │ TokenBudgeter（tiktoken）+ ContextMMU          │      │
│  └───────────────────────────────────────────────┘      │
│                                                         │
│  ┌─ 安全层 ─────────────────────────────────────┐      │
│  │ InputSanitizer + PermissionChecker + AuditLog │      │
│  └───────────────────────────────────────────────┘      │
│                                                         │
│  SQLite（12张表） · FAISS · BGE-M3 · React GUI           │
└────────────────────────────────────────────────────────┘
```

## 核心能力

| 类别 | 能力 |
|------|------|
| **多级记忆** | L0（输入）→ L1（对话缓存）→ L2（工作记忆）→ L3（长期记忆）→ L4（文件库） |
| **混合检索** | FAISS 语义向量 + FTS5 关键词 + 8 因子加权评分 |
| **代码理解** | tree-sitter AST 解析，提取函数/类/方法 + 签名 + docstring |
| **PDF 解析** | OpenDataLoader（结构化 Markdown/JSON）+ PyMuPDF 兜底 |
| **上下文装配** | 去重→排序→token 预算→来源标注→ContextPack |
| **任务图执行** | DAG 任务分解 + 拓扑排序调度 + 失败级联回退 |
| **多 Agent 协作** | Agent 注册表路由 + 共享黑板 + 6 阶段合并管线 |
| **真实 LLM** | OpenAI / DeepSeek / Anthropic 统一工厂（GUI 模型切换器） |
| **本地 Embedding** | BGE-M3（1024维，100+语言，纯 CPU） |
| **安全防护** | Prompt 注入检测、Agent 权限模型、审计日志 |
| **GitHub 索引** | GitPython 克隆 + 自动代码解析索引 |
| **可观测性** | 全链路执行追踪 + 结构化审计查询 |
| **评估体系** | 5 个评估场景 + precision@k/recall@k/MRR/nDCG/hit@k |

## 快速开始

### 环境要求

- Python ≥ 3.10
- Java ≥ 11（OpenDataLoader PDF 解析需要；可选，无 Java 时会自动回退 PyMuPDF）
- Node.js ≥ 18（GUI 构建需要）

### 安装

```bash
git clone https://github.com/reflectstars111/agentplan.git
cd agentplan
pip install -r requirements.txt

# 构建前端
cd gui && npm install && npm run build && cd ..
```

### 启动

```bash
# Mock 模式（无需 API Key，测试用）
python -m src

# DeepSeek + BGE-M3（推荐）
set DEEPSEEK_API_KEY=sk-你的key
python -m src --llm deepseek --model deepseek-chat

# OpenAI + BGE-M3
set OPENAI_API_KEY=sk-你的key
python -m src --llm openai --model gpt-4o

# 浏览器打开 http://127.0.0.1:8000
```

### 命令行参数

```
python -m src [选项]

  --llm    LLM 提供商       mock|openai|deepseek|anthropic（默认 mock）
  --model  模型名称          默认: deepseek-chat / gpt-4o
  --embed  Embedding 引擎    bge|mock（默认 bge）
  --port   服务端口          默认 8000
  --host   绑定地址          默认 127.0.0.1
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/upload` | 上传文本内容并索引 |
| `POST` | `/upload/file` | 上传文件（PDF/代码/Markdown） |
| `POST` | `/upload/github` | 克隆 GitHub 仓库并索引 |
| `POST` | `/query` | 简单问答（单 Agent 管线） |
| `POST` | `/task` | 任务图执行（多 Agent 管线） |
| `GET` | `/trace/{id}` | 查看执行追踪 |
| `GET` | `/health` | 健康检查 |

### 使用示例

```bash
# 上传知识库
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"content": "FastAPI 是一个现代 Python Web 框架...", "source_name": "fastapi.txt"}'

# 索引 GitHub 仓库
curl -X POST http://localhost:8000/upload/github \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'

# 简单问答
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是 FastAPI？", "model": "deepseek-chat"}'

# 任务图模式（复杂任务自动分解）
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"query": "分析这个项目的认证系统架构", "model": "deepseek-v4-pro"}'
```

## 项目结构

```
agentplan/
├── src/
│   ├── __main__.py           # 启动入口（python -m src）
│   ├── config.py             # 全局配置
│   ├── embedding.py          # BGE-M3 + OpenAI + mock 嵌入函数
│   ├── api/                  # FastAPI 应用 + 路由
│   ├── models/               # 10 个数据模型
│   ├── db/                   # SQLite 连接 + 迁移（12 张表）
│   ├── storage/              # FileStore、MemoryStore、Chunker、CodeChunker
│   ├── index/                # VectorIndex、KeywordIndex、StructureIndex、HybridRetriever
│   ├── context/              # TokenBudgeter、ContextMMU
│   ├── runtime/              # AgentRuntime、Controller、Scheduler、Verifier 等
│   ├── llm/                  # LLM 工厂 + Prompt 模板
│   ├── parsing/              # CodeParser（tree-sitter）、PDFParser
│   └── sources/              # GithubSource（clone + index）
├── gui/                      # React + Vite + TypeScript 前端
├── models/bge-m3/            # BGE-M3 本地模型文件
├── eval/                     # 评估场景 + 指标
├── tests/                    # 34 个测试文件，273 项测试
├── agent_os_initial_plan.md  # 原始架构设计文档
└── requirements.txt
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.13 |
| 数据库 | SQLite + FTS5 |
| 向量搜索 | FAISS（IndexFlatIP + IndexIDMap） |
| 本地 Embedding | BGE-M3（1024维，CPU 推理） |
| 代码解析 | tree-sitter（Python/JavaScript/TypeScript） |
| PDF 解析 | OpenDataLoader PDF + PyMuPDF |
| LLM 接入 | OpenAI / DeepSeek / Anthropic 统一工厂 |
| API 服务 | FastAPI + Uvicorn |
| 前端 | React 18 + TypeScript + Vite |
| 测试 | pytest（273 项，100% 通过） |
| Git 集成 | GitPython |

## 设计理念

Agent-OS 参考冯诺依曼计算机体系，将多 Agent 系统抽象为：

| 计算机体系 | Agent-OS 对应 |
|-----------|--------------|
| 运算器 | LLM 推理模块 |
| 存储器 | 多级上下文存储系统（L0-L5） |
| 控制器 | Controller / Scheduler / Context MMU |
| 页表 | 多级索引系统（FAISS + FTS5 + StructureIndex） |
| 缺页中断 | 上下文缺页机制（Context Page Fault） |
| 进程 | Agent 实例（AgentProcess） |
| 线程 | Agent 内部任务（Task） |
| 总线 | 共享黑板（SharedBlackboard） |

## 配置说明

`src/config.py` 核心配置项：

```python
# Embedding
embedding_dim: int = 1024          # BGE-M3 输出维度

# 检索权重（8 因子评分）
weight_semantic: float = 0.35      # 语义相似度
weight_keyword: float = 0.20       # 关键词匹配
weight_entity: float = 0.15        # 实体相关度
weight_recency: float = 0.10       # 时间新鲜度
weight_importance: float = 0.10    # 重要性
weight_structural: float = 0.10    # 结构相关度
penalty_token_cost: float = 0.10   # Token 成本惩罚
penalty_trust: float = 0.20        # 来源信任惩罚

# 上下文
default_token_budget: int = 24000  # 默认 token 预算
top_k_after_rerank: int = 15       # 重排序后返回数

# 任务执行
task_max_retries: int = 2          # 最大重试次数
parallel_enabled: bool = False     # 并行 Agent 执行
max_parallel_agents: int = 4       # 最大并行数

# 记忆写回门控
writeback_min_score: float = 0.5   # 最小写入分数
```

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 特定模块
python -m pytest tests/test_hybrid_retriever.py -v
python -m pytest tests/parsing/test_code_parser.py -v
```

273 项测试，100% 通过。

## 许可证

MIT
