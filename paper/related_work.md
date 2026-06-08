# Related Work: State Management for Long-Horizon AI Systems

## Survey Scope

We survey five categories of related work that address long-context and memory management for LLMs, comparing their approach with ARD's state-centric design.

---

## 1. OS-Inspired Memory Management: MemGPT / Letta

### Citation

> Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., & Gonzalez, J. E. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560. [https://arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)

### Key Idea

MemGPT treats the LLM as an **OS kernel** that actively manages its own memory hierarchy through function calls. It introduces a two-tier architecture:
- **Main Context** (analogous to RAM): system instructions + FIFO conversation queue
- **External Context** (analogous to disk): recall storage (full history) + archival storage (vector DB)

The LLM itself decides when to **page** information between tiers using function calls like `core_memory_replace()` and `archival_memory_insert()`.

### Evolution into Letta

As of September 2024, MemGPT evolved into the **Letta framework** (~17k GitHub stars):
- Introduced **Memory Blocks** — structured, labeled, editable units within the context window
- Introduced **"sleeptime compute"** — agents reflect and consolidate memories during idle periods
- Letta Filesystem achieved **74% accuracy on LoCoMo** with GPT-4o-mini, outperforming specialized memory tools like Mem0 (68.5%)
- **Benchmarking results**: Letta agents rank #4 overall (#1 OSS) on Terminal-Bench

### Comparison with ARD

| Dimension | MemGPT / Letta | ARD (Our Approach) |
|---|---|---|
| **Memory Model** | OS-inspired paging (LLM controls paging via function calls) | Database-inspired (Event Store + MVCC) |
| **Who manages memory?** | The LLM itself (via function calls) | The TransactionManager (deterministic optimistic locking) |
| **Consistency guarantee** | None — LLM decides paging heuristically | ACID-like (Transaction → Verify → Commit) |
| **Auditability** | Limited (conversation log only) | Complete (Trace Store + MVCC version history) |
| **Context strategy** | LLM self-manages what to keep/discard | Context MMU: 6-step deterministic pipeline |
| **Key difference** | Memory management is a **learned behavior** of the LLM | Memory management is a **system guarantee** provided by the runtime |

**Key insight for our work**: MemGPT validates the *need* for hierarchical memory in LLM systems. However, it delegates memory management to the LLM itself, making it probabilistic and non-deterministic. ARD provides deterministic, auditable memory management as a system-level guarantee.

> **Open question**: Does LLM-controlled paging (MemGPT) or system-controlled paging (ARD) produce more reliable long-horizon behavior? This is an empirical question directly relevant to H1.

---

## 2. Decoupled Memory Architecture: LongMem

### Citation

> Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., & Wei, F. (2023). *Augmenting Language Models with Long-Term Memory*. NeurIPS 2023. arXiv:2306.07174. [https://arxiv.org/abs/2306.07174](https://arxiv.org/abs/2306.07174)

### Key Idea

LongMem introduces a **decoupled network architecture** with three components:
1. **Frozen Backbone LLM**: Encodes historical context into a memory bank without parameter drift (solves the "memory staleness" problem)
2. **Residual Side-Network (SideNet)**: A lightweight trainable adapter that retrieves from memory and fuses with current context
3. **Cached Memory Bank**: Stores attention KV pairs from up to **65k tokens** of past context

Key innovation: **Decoupling memory encoding from memory reading** eliminates memory staleness and enables efficient adaptation without full model retraining.

### Results

- Perplexity improvement of -1.38 to -1.62 on long-text modeling
- **40.5% accuracy** on ChapterBreak (state-of-the-art at the time)
- With 2,000 cached examples, improved in-context learning performance

### Comparison with ARD

| Dimension | LongMem | ARD |
|---|---|---|
| **Memory mechanism** | Attention KV cache (low-level) | Semantic State Store (high-level) |
| **Training required?** | Yes (SideNet adapter training) | No (no model modification) |
| **State abstraction** | Raw attention keys/values | Structured state (User/Project/Task/Knowledge) |
| **Write capability** | Read-only (KV cache is side-effect of forward pass) | Transactional write with optimistic locking |
| **Versioning** | None | MVCC (seq_num = version) |

**Key insight for our work**: LongMem operates at the **attention mechanism level** (caching KV pairs), while ARD operates at the **semantic state level** (structured key-value state). LongMem is complementary — it could potentially serve as the underlying memory retrieval mechanism for ARD's Knowledge Store.

> **Open question**: Is attention-level memory (LongMem) or semantic state-level memory (ARD) more effective for maintaining consistency across long task sequences? This informs H3.

---

## 3. Memory-Augmented Retrieval: MemLong

### Citation

> Liu, W., Tang, Z., Li, J., Chen, K., & Zhang, M. (2024). *MemLong: Memory-Augmented Retrieval for Long Text Modeling*. arXiv:2408.16967. [https://arxiv.org/abs/2408.16967](https://arxiv.org/abs/2408.16967)

### Key Idea

MemLong uses an **external retriever** that extends context from 4K to **80K tokens** on a single NVIDIA 3090 GPU. Its innovation:

1. **Non-differentiable ret-mem module**: Retrieves historically relevant information from a memory bank
2. **Fine-grained retrieval attention**: Semantic-level chunk retrieval rather than raw KV cache
3. **Partial trainability**: Only a small portion of the model is trained

### Results

- **Extends context from 4k → 80k tokens** on consumer GPU
- Outperforms other LLMs on long-context benchmarks
- Computationally efficient (only partial model training)

### Comparison with ARD

| Dimension | MemLong | ARD |
|---|---|---|
| **Goal** | Extend context length efficiently | Manage context quality efficiently |
| **Approach** | Model-level (modify LLM attention) | System-level (no LLM modification) |
| **Memory type** | Retrieved text chunks (semantic) | State Store (structured) + Knowledge Store (chunks) |
| **Key difference** | **Longer** context (80K) | **Better** context (token budget management) |

**Key insight for our work**: MemLong and ARD share the premise that *retrieval is better than raw context extension*. But MemLong asks "how can we fit more into context?", while ARD asks "what is the **right** content to put in context?" This is directly relevant to H2: Context Window ≠ Memory, it's Execution Workspace.

> **Open question**: Given the same token budget, does ContextMMU's filtered context outperform MemLong's extended-but-unfiltered context in answer quality? This is the H3 experiment.

---

## 4. Hierarchical Retrieval: RAPTOR

### Citation

> Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*. arXiv:2401.18059. [https://arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059)

### Key Idea

RAPTOR builds a **hierarchical tree-structured datastore** through recursive summarization:

1. **Chunk** documents into short contiguous texts
2. **Embed** and **cluster** related chunks (GMM + UMAP)
3. **Summarize** each cluster using an LLM
4. **Repeat** recursively upward to build a multi-layer tree

At inference: retrieve from tree using either tree traversal or collapsed tree approach.

### Results

- **20% absolute improvement** on QuALITY benchmark with GPT-4 (over prior SOTA)
- State-of-the-art on NarrativeQA, QASPER, and QuALITY
- Outperformed both traditional retrieval and full-context models (including LongFormer)

### Comparison with ARD

| Dimension | RAPTOR | ARD |
|---|---|---|
| **Retrieval structure** | Recursive tree of summaries | Hybrid vector + keyword + entity + structure |
| **Summarization** | LLM-based recursive summarization | Token budget compression (Phase 1: truncation) |
| **State management** | None (stateless per-query retrieval) | StateStore with transactional writeback |
| **Cross-document** | Implicit (summaries span clusters) | Explicit (knowledge store spans sources) |
| **Multi-turn** | No support | Yes (StateStore loads prior session state) |

**Key insight for our work**: RAPTOR's hierarchical summarization is the closest parallel to ContextMMU's COMPRESS step. But RAPTOR is purely retrieval-focused and stateless — it has no concept of state accumulation across queries. ARD extends this with transactional writeback.

> **Open question**: Could RAPTOR-style recursive summarization be integrated into ContextMMU's COMPRESS step to improve compression quality?

---

## 5. Advanced RAG Baselines

### 5.1 Naive RAG (Baseline 1 in our experiments)

> Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

**Approach**: Query → Vector Search → Top-K chunks → Concatenate → LLM.

**Limitations**: Single retrieval strategy, no filtering, no budget management, no state. All chunks enter context regardless of relevance.

### 5.2 HyDE (Hypothetical Document Embeddings)

> Gao, L., et al. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels*. arXiv:2212.10496. [https://arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496)

**Approach**: Generate a hypothetical answer document from the query → embed the hypothetical → use it to retrieve real documents.

**Limitations**: Improves retrieval relevance but adds latency (LLM call before retrieval). No state management.

### 5.3 Self-RAG

> Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024. [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)

**Approach**: LLM trained to decide *when* to retrieve, *what* to retrieve, and *how* to critique its own output. Reflection tokens signal retrieval necessity and answer quality.

**Limitations**: Requires special training (reflection tokens). Critique is learned behavior, not guaranteed. No persistent state across queries.

### 5.4 Corrective RAG (CRAG)

> Yan, S., et al. (2024). *Corrective Retrieval Augmented Generation*. arXiv:2401.15884. [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)

**Approach**: Evaluates retrieval quality first. If retrieved documents are low-quality, triggers web search as fallback. Decomposes then recomposes for complex queries.

**Limitations**: Focuses on retrieval quality only. Does not address context assembly, budget management, or state persistence.

---

## 6. Cross-Cutting Themes and ARD's Positioning

### Theme 1: Who Controls Memory?

| System | Control Mechanism | Deterministic? |
|---|---|---|
| MemGPT/Letta | LLM function calls | ❌ Probabilistic |
| LongMem | Trained SideNet adapter | ⚠️ Learned (deterministic per input) |
| MemLong | Trained ret-mem module | ⚠️ Learned |
| RAPTOR | Algorithmic clustering | ✅ Deterministic |
| HyDE | LLM hallucination → retrieval | ❌ Probabilistic |
| Self-RAG | LLM reflection tokens | ❌ Probabilistic |
| **ARD** | **TransactionManager + Optimistic Locking** | **✅ Deterministic** |

**ARD's advantage**: Memory write-back is a system guarantee, not a learned behavior. The TransactionManager verifies consistency before every commit. This is the core of H1.

### Theme 2: Context = Memory or Execution Workspace?

| System | Context Model | Budget Control |
|---|---|---|
| Naive RAG | Context = dump all retrieved chunks | No budget |
| MemGPT | Context = RAM (actively managed) | LLM self-manages |
| RAPTOR | Context = tree traversal result | Implicit (summary depth) |
| MemLong | Context = retrieval-augmented input | Model-determined |
| **ARD** | **Context = Execution Workspace** | **6-step pipeline with explicit token budget** |

**ARD's advantage**: ContextMMU treats context as a precious execution resource with explicit budget allocation. This is the core of H2.

### Theme 3: State Across Time?

| System | State Across Queries | Versioning | Rollback |
|---|---|---|---|
| Naive RAG | ❌ Stateless | ❌ | ❌ |
| MemGPT/Letta | ✅ Conversation + archival memory | ❌ | ❌ (manual edits) |
| LongMem | ❌ Read-only KV cache | ❌ | ❌ |
| RAPTOR | ❌ Stateless | ❌ | ❌ |
| **ARD** | **✅ StateStore** | **✅ MVCC (seq_num)** | **✅ Transaction Rollback** |

**ARD's advantage**: ARD is the only system that provides true, auditable, versioned state across queries with rollback capability. This is the core of H3.

---

## 7. Research Gap

The survey reveals a clear gap that ARD addresses:

> **No existing system combines (a) deterministic state management with (b) explicit context budget control and (c) ACID-like transactional writeback with MVCC versioning.**

| System | Deterministic State | Budget Control | Transactional Write | MVCC |
|---|---|---|---|---|
| MemGPT/Letta | ❌ | ⚠️ (LLM-controlled) | ❌ | ❌ |
| LongMem | ✅ | ❌ | ❌ | ❌ |
| MemLong | ⚠️ | ❌ | ❌ | ❌ |
| RAPTOR | ✅ | ⚠️ (implicit) | ❌ | ❌ |
| Self-RAG | ❌ | ⚠️ (critique-based) | ❌ | ❌ |
| **ARD** | **✅** | **✅** | **✅** | **✅** |

This gap forms the primary contribution claim of our work.

---

## 8. References

1. Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
2. Wang, W. et al. (2023). Augmenting Language Models with Long-Term Memory. NeurIPS 2023. arXiv:2306.07174.
3. Liu, W. et al. (2024). MemLong: Memory-Augmented Retrieval for Long Text Modeling. arXiv:2408.16967.
4. Sarthi, P. et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv:2401.18059.
5. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
6. Gao, L. et al. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels. arXiv:2212.10496.
7. Asai, A. et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR 2024. arXiv:2310.11511.
8. Yan, S. et al. (2024). Corrective Retrieval Augmented Generation. arXiv:2401.15884.
