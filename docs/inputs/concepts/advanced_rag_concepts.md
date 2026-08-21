# 10 Advanced RAG Concepts — Full Extraction, Combinations & Compatibility Matrix

> **Source:** Instagram Reel by @darpan.decoded  
> **Topic:** Advanced Retrieval-Augmented Generation (RAG) Techniques  
> **Concepts Covered:** 9 Core RAG Concepts + Bonus Interview Questions

---

## Table of Contents
1. [Concept Extraction](#1-concept-extraction)
2. [Combination Matrix & Pipeline Archetypes](#2-combination-matrix--pipeline-archetypes)
3. [Full Compatibility Analysis](#3-full-compatibility-analysis)
4. [How Every Combination Works](#4-how-every-combination-works)
5. [Implementation Roadmap](#5-implementation-roadmap)

---

## 1. Concept Extraction

### Concept 1: Reranking
**Tagline:** *Retrieve brings candidates. Reranking finds the best.*

**The Problem:**  
Your vector DB searched through 100 pages and returned 20 chunks. But what if the BEST answer is ranked #19? LLMs usually see only the top 3-5 chunks. The right answer may never reach the model.

**How It Works:**
- **Without Reranking:** Vector Search → 20 Chunks → LLM sees top 3-5 only → Best answer at #19 gets ignored.
- **With Reranking:** Vector Search → 20 Chunks → Reranker sorts by relevance → Top 5 Best Chunks → LLM gets the most relevant context. Best answer moves to the top.

**What Does a Reranker Do?**
- Looks at the query and each retrieved chunk together
- Answers: *"Which chunks actually answer the question best?"*
- Understands context better, finds hidden relevance, filters out noise, improves accuracy

**Types of Rerankers:**
| Type | Accuracy | Speed | Cost |
|------|----------|-------|------|
| Cross-Encoder | High | Slower | Medium |
| Bi-Encoder + Rerank Model | Balanced | Balanced | Balanced |
| LLM-based Reranker | Highest | Slowest | Costlier |

**Example:**  
Query: *"How to deploy FastAPI in production?"*
- Without rerank: Chunk about logging comes at #1
- With rerank: Chunk about Gunicorn + Nginx comes at #1

**Key Takeaway:** Retrieve more. Rerank smart. Let LLM see the BEST, not just the first.

---

### Concept 2: Hybrid Search
**Tagline:** *Meaning + Keywords. That's how you find more of the RIGHT stuff.*

**The Problem:**  
Sometimes meaning matters. Sometimes exact words matter. Relying on only one search means you MISS good results.

**Reality Check:** Users don't always search perfectly; docs use different words; spelling/synonyms/abbreviations happen. You need BOTH.

**Formula:**  
`Hybrid Search = Vector (understands meaning) + BM25 (understands keywords) = Better Results`

**How It Works:**
1. **Vector Search** (Semantic) → Top 5 Results
2. **Keyword Search** (BM25) → Top 5 Results
3. **Merge & Rank** (RRF / Weighted Scoring) → Final Top Results

**When Hybrid Search Shines:**
- Technical terms + natural language mix
- Spelling mistakes / abbreviations
- Docs with inconsistent wording
- Need high recall + high precision

**Popular Combination Methods:** Reciprocal Rank Fusion (RRF), Weighted Score Combination, Relative Score Fusion, Rank Based Fusion

**Key Takeaway:** Don't choose between meaning or keywords. Use BOTH.

---

### Concept 3: Chunking Strategies
**Tagline:** *Good chunks = Better Retrieval = Better Answers*

**Why Chunking Matters:**
- Too big → lost in the middle, less precise
- Too small → more noise, poor context
- Right chunk → more relevant, more useful

**Golden Rule:**  
*Keep chunks as small as possible to be precise, but as large as necessary to be complete.*

**6 Popular Chunking Strategies:**

| # | Strategy | Description | Pros | Cons |
|---|----------|-------------|------|------|
| 1 | **Fixed Size Chunking** | Split into fixed length (e.g., 500 tokens) with overlap | Simple & fast | May cut sentences |
| 2 | **Sentence Based** | Split at sentence boundaries | Preserves meaning | Length varies a lot |
| 3 | **Semantic Chunking** | Group by semantic similarity | High relevance, topic coherent | Complex, needs embeddings |
| 4 | **Parent Document Chunking** | Small chunks for retrieval, return larger parent | More context | Higher token usage |
| 5 | **Sliding Window** | Window slides step by step | Good coverage, keeps flow | More storage |
| 6 | **Structure Aware** | Use headings, lists, tables, code blocks | Respects structure | Needs good parser |

**Tips:** Understand your data, test sizes (256, 512, 1024), use 10-20% overlap, evaluate with metrics.

**Common Practice:** Hybrid Approach = Structure aware + semantic grouping + parent document retrieval.

**Key Takeaway:** Good chunks bring the right context. Right context helps the LLM give the right answer.

---

### Concept 4: Multi Query Retrieval
**Tagline:** *One question. Multiple perspectives. Better results.*

**The Problem:**  
Even with Query Expansion, still searching in one direction. What if the answer is written completely differently? One query → limited view → might still MISS!

**How It Works:**
1. **Original Question**
2. **LLM Creates Multiple Queries** (diverse viewpoints)
3. **Search All Queries** (vector search each)
4. **Merge & Rerank** → Final Top Chunks

**Multi Query vs Query Expansion:**
| Feature | Query Expansion | Multi Query |
|---------|----------------|-------------|
| Focus | Better wording | Different viewpoints |
| Queries | Similar | Diverse |
| Recall | Good | Much higher |
| Use case | General | Production RAG |

**When It Shines:** Large/messy KBs, tech docs with varied terminology, enterprise/multi-domain docs, research/legal/medical.

**Popular Implementations:** LlamaIndex MultiQueryRetriever, LangChain MultiQueryRetriever, Custom LLM prompt retriever.

**Key Takeaway:** Don't ask once. Ask in multiple smart ways. More angles → More knowledge.

---

### Concept 5: Parent Document Retrieval
**Tagline:** *Small chunks = better search. Parent docs = better understanding.*

**The Problem:**  
Retrieve a tiny chunk that looks relevant but misses full context. The chunk by itself doesn't make sense. Half the story ≠ The truth.

**How It Works:**
1. **Chunk Creation** → Small chunks for better search
2. **Vector Search** → Get top relevant chunks
3. **Map to Parent** → Find parent document/page/section
4. **Send to LLM** → Full context, not just tiny piece

**Example:**  
Query: *"Why did FastAPI become so popular?"*
- Chunk (out of context): *"...it is fast, easy to use, and has automatic docs..."* ❌
- Parent Doc: *"...FastAPI became popular because it is fast, easy to use, has automatic docs, async support, and great developer experience..."* ✅

**When to Use:** Very small chunks (100-300 tokens), docs with lots of references, answers need full context.

**Tips:** Store parent_id, use page/section level, avoid too many parents, balance size vs relevance.

**Bottom Line:** Chunks help you FIND. Parent docs help the model UNDERSTAND. Both together give the BEST answers.

---

### Concept 6: Context Compression
**Tagline:** *Too much context = lost in the middle. Compress it. Keep what matters.*

**The Problem:**  
Retriever found 40 chunks. LLM can only take 8. Send everything → model gets distracted, important info buried, quality drops.

**How It Works:**
1. **Many Retrieved Chunks** (irrelevant, repetitive, noisy)
2. **Compress** → Remove noise, duplicates, filler
3. **Keep What Matters** → High relevance, unique info, answer focused, clean & concise
4. **Send to LLM** → Clear, focused context = better answers

**What Gets Compressed:** Duplicate chunks, filler sentences, low relevance, long-winded details, off-topic parts.

**Popular Ways:** LLM Summarization, Keyword/Keyphrase extraction, Redundancy Removal, Extractive Compression, Relevance Scoring.

**Example:** 40 chunks → 8 chunks → Token usage -75% → Answer Quality ↑

**Tips:** Tune compression level, measure answer quality not just tokens, compression > bigger models, small clean context beats huge noisy context.

**Key Takeaway:** More context is not always better. Relevant context is. Compress smartly.

---

### Concept 7: HyDE (Hypothetical Document Embeddings)
**Tagline:** *Sometimes the question is too vague for good search. So we first imagine a perfect answer, then search using that.*

**The Problem:**  
Question is too short or vague. Direct search misses best docs because query lacks right words. Vague query → Weak search → Poor results.

**How HyDE Works:**
1. **User Question** (short, vague)
2. **Generate Hypothetical Answer** (LLM writes ideal answer)
3. **Create Embedding** (embed hypothetical answer, NOT the question)
4. **Vector Search** (powerful embedding = better match)
5. **Retrieve Relevant Docs**
6. **Send to LLM**

**Not Just Guessing:** Generates structured, relevant, domain-aware hypothetical answer. Captures meaning you intended but failed to express.

**Example:** Query: *"FastAPI background tasks"*
- Direct Search: Celery, threading, cron jobs, async jobs ❌
- HyDE Search: BackgroundTasks, add_task, response, execution ✅

**Why It Works:** Uses more meaningful terms, captures context/intent, improves recall significantly, great for vague/complex questions.

**Tips:** Use good LLM, keep generation concise but rich, works better with Reranking, combine with Hybrid Search for max recall.

**Key Takeaway:** HyDE turns a weak question into a strong search signal. Better signal → Better retrieval → Better answers.

---

### Concept 8: Self-RAG (Self Reflective RAG)
**Tagline:** *Why search every time? Let the model decide first.*

**The Problem:**  
In normal RAG, always retrieve even for simple questions. Wastes time, tokens, money. What if the model already knows?

**How Self-RAG Works:**
1. **User Question**
2. **Need Retrieval?** → LLM checks its own knowledge, decides YES or NO
   - **NO** → Answer Directly (internal knowledge)
   - **YES** → Retrieve from Vector DB
3. **Generate Answer**

**Example:**
- Q1: *"What is 2 + 2?"* → I know this → Direct answer ✅
- Q2: *"Latest FastAPI release notes?"* → Not sure → Retrieve → Answer ✅

**Benefits:** Reduces unnecessary retrievals, saves tokens & cost, faster responses, less load on Vector DB.

**When "NO":** General knowledge, math/logic/facts, common sense, confident questions.
**When "YES":** Recent/live info, domain-specific knowledge, private docs, complex multi-hop questions.

**Key Takeaway:** Self-RAG doesn't just retrieve... it thinks first. Smarter decisions → Better efficiency → Better answers.

---

### Concept 9: CRAG (Corrective RAG)
**Tagline:** *Not every retrieved chunk is useful. CRAG checks the quality before trusting it.*

**The Problem:**  
Retriever is not perfect. Can return irrelevant, outdated, or misleading chunks. If LLM trusts bad context → wrong answers with full confidence.

**How CRAG Works:**
1. **User Question**
2. **Retrieve** → top-k documents
3. **Evaluate** → Relevant? Complete? Consistent? Support the query?
4. **Decision:**
   - **GOOD** → Use these docs
   - **BAD** → Need correction
5. **Correct / Retrieve Again** → Refine query, alternative search, expand/re-rank, filter noise
6. **Final Answer** → Only high-quality context reaches LLM

**CRAG = Retrieve → Check → Correct → Answer**

**Example:** Query: *"Best database for real-time analytics?"*  
Retriever returns: SQLite tutorial, MongoDB basics (outdated), ClickHouse case study, Postgres setup.  
CRAG checks → removes weak docs → fetches better → accurate answer.

**Benefits:** Prevents wrong/confident answers, improves accuracy, reduces hallucinations, handles complex queries, saves tokens.

**When CRAG Helps Most:** Complex/multi-hop questions, many interpretations, low retriever quality, domain-specific search, ambiguous queries.

**Tips:** Use strong evaluator (LLM/cross-encoder), clear quality criteria, rejection threshold, smart re-query strategy, combine with Reranking & Hybrid Search.

**Key Takeaway:** CRAG doesn't just trust the retriever... It verifies, corrects, ensures the best context. Better validation → Better retrieval → Better answers.

---

### Bonus: 10 Interview Questions

| # | Interview Question | What They're Testing |
|---|-------------------|---------------------|
| 1 | RAG retrieves relevant docs but LLM gives incorrect answers. How to identify if problem is retrieval, context construction, or generation? | RAG debugging, observability, failure attribution |
| 2 | How to measure quality of production RAG? Which metrics for retrieval vs final answer? | Retrieval metrics, faithfulness, relevance, groundedness |
| 3 | Top-5 chunks are relevant but LLM performs worse than with only 2 chunks. Why? How to fix? | Context noise, lost-in-the-middle, reranking, compression |
| 4 | Design RAG pipeline when users use terminology never appearing literally in docs | Semantic retrieval, query rewriting, HyDE, multi-query |
| 5 | 10,000 docs, vector search returns semantically similar but wrong docs. Improve precision without increasing top_k? | Metadata filtering, hybrid search, reranking, indexing |
| 6 | Handle questions requiring info from multiple documents/chunks | Multi-hop retrieval, query decomposition, iterative retrieval |
| 7 | Retriever returns low-confidence/irrelevant results. Should LLM still answer? | Retrieval confidence, abstention, CRAG/Self-RAG, hallucination prevention |
| 8 | Choose chunk size and overlap for production RAG. What if too small or too large? | Chunking strategy, semantic boundaries, retrieval granularity |
| 9 | System works on 100 docs but drops at 1M docs. What to rethink? | Scalability, indexing, partitioning, hybrid retrieval, latency |
| 10 | Prevent LLM from answering from own knowledge when context doesn't support it | Grounding, citations, prompt constraints, verification |

> **Final Wisdom:** Think like a system. Validate like a scientist. Ship like an engineer.

---

## 2. Combination Matrix & Pipeline Archetypes

### 2.1 Single Concepts (9 combinations)

Each concept improves a specific pipeline stage:

| Concept | Standalone Value | Pipeline Stage |
|---------|-----------------|----------------|
| Reranking | Improves precision of retrieved results | Post-Retrieval |
| Hybrid Search | Improves recall (semantic + keyword) | Retrieval |
| Chunking Strategies | Foundational — determines retrieval quality | Pre-Processing |
| Multi Query Retrieval | Improves recall via query diversification | Pre-Retrieval |
| Parent Document Retrieval | Improves context completeness | Post-Retrieval |
| Context Compression | Reduces noise and token usage | Post-Retrieval |
| HyDE | Improves query representation | Pre-Retrieval |
| Self-RAG | Reduces unnecessary retrievals | Pre-Retrieval / Routing |
| CRAG | Ensures retrieval quality via validation | Post-Retrieval |

---

### 2.2 Pair Combinations (36 combinations)


| Pair | Synergy | How They Work Together | Score |
|------|---------|----------------------|-------|
| **Reranking + Hybrid Search** | ⭐⭐⭐⭐⭐ | Hybrid brings diverse candidates from semantic + keyword spaces; Reranking precisely orders them by true relevance. | **A+** |
| **Reranking + Multi Query** | ⭐⭐⭐⭐⭐ | Multi Query generates diverse pools; Reranking deduplicates and sorts merged results. | **A+** |
| **Reranking + HyDE** | ⭐⭐⭐⭐⭐ | HyDE generates strong hypothetical doc; Reranking validates retrieved chunks match ORIGINAL query. | **A+** |
| **Reranking + Parent Document** | ⭐⭐⭐⭐⭐ | Reranker scores small chunks; Parent Document swaps in full parent context for top-ranked chunks. | **A+** |
| **Reranking + Context Compression** | ⭐⭐⭐⭐☆ | Reranking ensures only best chunks enter compression; prevents good chunks being compressed with noise. | **A** |
| **Reranking + CRAG** | ⭐⭐⭐⭐⭐ | CRAG evaluates quality; if rejected, Reranking reapplied after corrective loop. Double-layer QA. | **A+** |
| **Reranking + Self-RAG** | ⭐⭐⭐☆☆ | Self-RAG may skip retrieval; when retrieval happens, Reranking improves it. No conflict. | **B+** |
| **Reranking + Chunking** | ⭐⭐⭐⭐☆ | Better chunking → better inputs for Reranking. Foundational synergy. | **A** |
| **Hybrid Search + Multi Query** | ⭐⭐⭐⭐⭐ | Each of N queries runs through BOTH vector and keyword search. Massive recall boost. | **A+** |
| **Hybrid Search + HyDE** | ⭐⭐⭐⭐⭐ | HyDE hypothetical answer provides rich signal for BOTH semantic embedding AND keyword extraction. | **A+** |
| **Hybrid Search + Parent Document** | ⭐⭐⭐⭐☆ | Hybrid finds right small chunks; Parent expands them. Keyword component good at finding exact refs. | **A** |
| **Hybrid Search + Context Compression** | ⭐⭐⭐☆☆ | Hybrid may return more results; Compression handles volume. Functional. | **B+** |
| **Hybrid Search + CRAG** | ⭐⭐⭐⭐⭐ | CRAG evaluates results from vector and keyword arms separately. Detects underperforming arm. | **A** |
| **Hybrid Search + Self-RAG** | ⭐⭐⭐☆☆ | Self-RAG decides IF; Hybrid Search decides HOW. Orthogonal, no conflict. | **B+** |
| **Hybrid Search + Chunking** | ⭐⭐⭐⭐☆ | Structure-aware chunking improves both semantic and keyword search quality. | **A** |
| **Multi Query + HyDE** | ⭐⭐⭐⭐⭐ | HyDE as one strong query; Multi Query generates multiple phrasings. Combine both approaches. | **A+** |
| **Multi Query + Parent Document** | ⭐⭐⭐⭐☆ | Multiple queries find chunks from different angles; Parent ensures each brings full context. | **A** |
| **Multi Query + Context Compression** | ⭐⭐⭐⭐☆ | Multi Query inflates candidate pool; Compression deflates to manageable set. Natural equilibrium. | **A** |
| **Multi Query + CRAG** | ⭐⭐⭐⭐⭐ | CRAG evaluates each query's result set independently. Identifies bad angles and drops/re-queries them. | **A+** |
| **Multi Query + Self-RAG** | ⭐⭐⭐☆☆ | Self-RAG may bypass Multi Query for known questions. When retrieval needed, Multi Query shines. | **B+** |
| **Multi Query + Chunking** | ⭐⭐⭐⭐☆ | Semantic chunking ensures different query variations map to same conceptual chunks. | **A** |
| **HyDE + Parent Document** | ⭐⭐⭐⭐☆ | HyDE finds right semantic neighborhood; Parent ensures LLM gets complete context. | **A** |
| **HyDE + Context Compression** | ⭐⭐⭐☆☆ | HyDE improves retrieval quality; Compression handles volume. Compatible. | **B+** |
| **HyDE + CRAG** | ⭐⭐⭐⭐⭐ | CRAG validates docs retrieved via HyDE actually answer ORIGINAL question. Critical safety check. | **A+** |
| **HyDE + Self-RAG** | ⭐⭐⭐☆☆ | Self-RAG may skip retrieval. Can use Self-RAG to activate HyDE only for vague queries. | **B+** |
| **HyDE + Chunking** | ⭐⭐⭐⭐☆ | Semantic chunks align with topic clusters HyDE's hypothetical answer maps to. | **A** |
| **Parent Document + Context Compression** | ⭐⭐⭐⭐☆ | Parent expands chunks (more tokens); Compression trims expanded parents. Perfect balance. | **A** |
| **Parent Document + CRAG** | ⭐⭐⭐⭐⭐ | CRAG evaluates PARENT documents (not just chunks) for quality. Validates completeness. | **A+** |
| **Parent Document + Self-RAG** | ⭐⭐⭐☆☆ | Self-RAG decides whether to retrieve; Parent determines what to return. Compatible. | **B+** |
| **Parent Document + Chunking** | ⭐⭐⭐⭐⭐ | Parent Document Chunking (Strategy #4) IS Parent Document Retrieval at chunking layer. Deeply intertwined. | **A+** |
| **Context Compression + CRAG** | ⭐⭐⭐⭐⭐ | CRAG filters bad docs BEFORE compression. Prevents "compressing garbage." | **A+** |
| **Context Compression + Self-RAG** | ⭐⭐⭐☆☆ | Self-RAG may bypass retrieval. When retrieval happens, Compression helps. | **B+** |
| **Context Compression + Chunking** | ⭐⭐⭐⭐☆ | Smaller semantic chunks reduce need for aggressive compression. Less info loss. | **A** |
| **CRAG + Self-RAG** | ⭐⭐⭐⭐⭐ | Self-RAG decides IF to retrieve; CRAG evaluates WHAT was retrieved. Perfect complement. | **A+** |
| **CRAG + Chunking** | ⭐⭐⭐⭐☆ | CRAG's evaluator works better with coherent, complete chunks. | **A** |
| **Self-RAG + Chunking** | ⭐⭐⭐☆☆ | Self-RAG is query-dependent; chunking is data-dependent. Independent improvements. | **B+** |

---

### 2.3 Triple Combinations (84 combinations) — Key Archetypes

#### Archetype A: The "Maximum Recall" Pipeline
**Components:** Hybrid Search + Multi Query + HyDE

**How it works:**
1. User asks vague question
2. **HyDE** generates rich hypothetical answer → strong embedding
3. **Multi Query** generates 5 diverse phrasings
4. Each query + HyDE embedding fed into **Hybrid Search** (vector + BM25)
5. Results merged from all angles
6. → Unmatched recall. Finds docs any single method would miss.

**Best for:** Research assistants, legal/medical search, enterprise KBs with messy data.

---

#### Archetype B: The "Precision Fortress" Pipeline
**Components:** Reranking + CRAG + Parent Document Retrieval

**How it works:**
1. Initial retrieval returns candidate chunks
2. **Reranking** scores chunks by true relevance → top chunks identified
3. **Parent Document Retrieval** expands top chunks into full parent sections
4. **CRAG** evaluates expanded parents: Relevant? Complete? Consistent?
5. If CRAG rejects → trigger corrective retrieval loop
6. Only validated, high-quality, complete context reaches LLM

**Best for:** Financial AI, medical diagnosis, legal research — domains where wrong answers are costly.

---

#### Archetype C: The "Efficiency Master" Pipeline
**Components:** Self-RAG + Context Compression + Reranking

**How it works:**
1. **Self-RAG** checks if LLM already knows the answer
2. If YES → direct answer (zero retrieval cost)
3. If NO → retrieve candidates
4. **Reranking** quickly surfaces best chunks
5. **Context Compression** distills top chunks into tight, focused context
6. → Minimal tokens, minimal latency, maximum accuracy

**Best for:** Customer support chatbots, high-traffic AI assistants, cost-sensitive production systems.

---

#### Archetype D: The "Complete Context" Pipeline
**Components:** Parent Document Retrieval + Context Compression + Hybrid Search

**How it works:**
1. **Hybrid Search** finds relevant small chunks (semantic + keyword)
2. **Parent Document Retrieval** maps each chunk to full parent section
3. **Context Compression** summarizes expanded parents, removing redundancy
4. LLM gets complete context without token bloat

**Best for:** Long-document Q&A (PDFs, books, technical manuals), code documentation.

---

#### Archetype E: The "Adaptive Intelligence" Pipeline
**Components:** Self-RAG + CRAG + Multi Query

**How it works:**
1. **Self-RAG** decides if retrieval is needed
2. If YES → **Multi Query** generates diverse search angles
3. Results retrieved from multiple query variations
4. **CRAG** evaluates combined result set for quality
5. If quality low → CRAG triggers re-query with refined angles
6. → System adapts retrieval strategy based on question complexity

**Best for:** General-purpose AI assistants, research tools, complex Q&A systems.

---

#### Archetype F: The "Semantic Powerhouse" Pipeline
**Components:** HyDE + Semantic Chunking + Reranking

**How it works:**
1. Documents split using **Semantic Chunking** (topic-coherent clusters)
2. User asks vague question → **HyDE** generates hypothetical answer
3. Hypothetical answer's embedding maps cleanly to semantic chunk clusters
4. **Reranking** validates retrieved clusters truly answer the query
5. → Highly coherent, topic-aligned retrieval

**Best for:** Content recommendation, research paper search, topic-based knowledge bases.

---

#### Archetype G: The "Production Grade" Pipeline
**Components:** Hybrid Search + Reranking + CRAG + Context Compression

**How it works:**
1. **Hybrid Search** casts wide net (semantic + keyword)
2. **Reranking** narrows to truly relevant candidates
3. **CRAG** validates quality and triggers correction if needed
4. **Context Compression** optimizes final context for LLM
5. → Gold standard for production RAG systems

**Best for:** Enterprise search, production chatbots, any system where reliability matters.

---

### 2.4 Quadruple+ Combinations — Advanced Pipeline Patterns

#### Pattern 1: The "Fort Knox" RAG (6 concepts)
**Components:** Hybrid Search + Multi Query + HyDE + Reranking + CRAG + Parent Document Retrieval

**Flow:**
```
User Query
    ↓
HyDE generates hypothetical answer
    ↓
Multi Query generates 5 diverse queries
    ↓
Each query → Hybrid Search (Vector + BM25)
    ↓
Merge all results
    ↓
Reranking scores all candidates
    ↓
Top-K chunks → Map to Parent Documents
    ↓
CRAG evaluates parent docs for quality
    ↓
[If BAD] → Corrective retrieval loop
    ↓
[If GOOD] → Send complete, validated context to LLM
    ↓
High-quality answer
```
**Trade-offs:** High latency, high cost, unbeatable accuracy.

---

#### Pattern 2: The "Speed Demon" RAG (4 concepts)
**Components:** Self-RAG + Hybrid Search + Reranking + Context Compression

**Flow:**
```
User Query
    ↓
Self-RAG: "Do I know this?"
    ↓ YES → Direct answer (fastest path)
    ↓ NO  → Continue
Hybrid Search (quick parallel vector + keyword)
    ↓
Reranking (lightweight cross-encoder)
    ↓
Context Compression (extractive, fast)
    ↓
Answer
```
**Trade-offs:** Optimized for speed and cost. Slightly lower recall than Fort Knox.

---

#### Pattern 3: The "Research Assistant" RAG (5 concepts)
**Components:** Multi Query + HyDE + Hybrid Search + Parent Document + Context Compression

**Flow:**
```
Complex research question
    ↓
HyDE generates domain-aware hypothetical answer
    ↓
Multi Query creates 5+ academic phrasings
    ↓
Hybrid Search across abstracts + full text
    ↓
Parent Document Retrieval gets full papers for top abstracts
    ↓
Context Compression extracts key findings
    ↓
Synthesized research answer
```
**Trade-offs:** Optimized for recall and completeness. Higher token usage acceptable.

---

#### Pattern 4: The "All-In" RAG (All 9 concepts)
**Components:** Chunking + Hybrid Search + Multi Query + HyDE + Reranking + Parent Document + Compression + Self-RAG + CRAG

**Flow:**
```
Documents → Structure-Aware Semantic Chunking (with overlap)
    ↓
User Query
    ↓
Self-RAG: "Need retrieval?"
    ↓ NO → Direct answer
    ↓ YES →
HyDE generates hypothetical answer
    ↓
Multi Query generates diverse phrasings
    ↓
All queries → Hybrid Search (Vector + BM25)
    ↓
Reranking scores merged results
    ↓
Top chunks → Parent Document Retrieval
    ↓
CRAG evaluates expanded parents
    ↓
[If BAD] → Corrective loop (refine queries, re-rank)
    ↓
[If GOOD] → Context Compression
    ↓
Final, validated, compressed context → LLM
    ↓
Answer
```
**Trade-offs:** Maximum complexity. Only justified for mission-critical applications.

---

## 3. Full Compatibility Analysis

### 3.1 Compatibility Matrix

|  | Rerank | Hybrid | Chunking | MultiQ | Parent | Compress | HyDE | Self-RAG | CRAG |
|--|:------:|:------:|:--------:|:------:|:------:|:--------:|:----:|:--------:|:----:|
| **Rerank** | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Hybrid** | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Chunking** | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MultiQ** | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Parent** | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| **Compress** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ |
| **HyDE** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| **Self-RAG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **CRAG** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |

**Legend:** ✅ = Fully Compatible (all pairs work together)

**Important:** ALL 9 concepts are mutually compatible. There are **zero conflicts**. Only constraints are latency, cost, and complexity budgets.

---

### 3.2 Compatibility by Pipeline Stage

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE STAGES                              │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│  PRE-PROC   │ PRE-RETRIEVE│   RETRIEVE  │ POST-RETRIEVE│   GENERATION   │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│ • Chunking  │ • Self-RAG  │ • Hybrid    │ • Reranking │ • LLM Answer   │
│  Strategies │   (decision)│   Search    │ • Parent    │                │
│             │ • Multi     │             │   Document  │                │
│             │   Query     │             │ • Context   │                │
│             │ • HyDE      │             │   Compress  │                │
│             │             │             │ • CRAG      │                │
│             │             │             │   (evaluate)│                │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘
```

**Compatibility Rules:**

| Rule | Explanation |
|------|-------------|
| **Within-stage** | Concepts in same stage are alternatives OR can be chained. Example: Multi Query AND HyDE both run in Pre-Retrieve. |
| **Adjacent-stage** | Always synergistic. Output of one stage feeds into next. Example: Hybrid Search → Reranking. |
| **Skip-stage** | Still compatible but reduced synergy. Example: Chunking (Pre-Proc) + Self-RAG (Pre-Retrieve). |
| **Feedback-loop** | CRAG can feed back to ANY earlier stage. This is its superpower. |

---

### 3.3 Synergy Heatmap

```
                Rnk  Hyb  Chk  Mul  Par  Cmp  Hyd  SfR  CRG
Reranking      [█]  ███  ██  ███  ███  ███  ███  ██  ███
Hybrid Search   ███ [█]  ██  ███  ███  ██  ███  ██  ███
Chunking        ██   ██  [█]  ██  ███  ██  ██  ██  ██
Multi Query     ███  ███  ██  [█]  ███  ███  ███  ██  ███
Parent Doc      ███  ███  ███  ███  [█]  ███  ███  ██  ███
Compression     ███  ██  ██  ███  ███  [█]  ██  ██  ███
HyDE            ███  ███  ██  ███  ███  ██  [█]  ██  ███
Self-RAG        ██   ██  ██  ██  ██  ██  ██  [█]  ███
CRAG            ███  ███  ██  ███  ███  ███  ███  ███  [█]

Legend: ███ = Strong synergy | ██ = Moderate synergy | [█] = Self
```

---

### 3.4 Conflict Analysis

| Potential Conflict | Reality | Verdict |
|-------------------|---------|---------|
| Self-RAG skips retrieval → makes Multi Query/HyDE/Hybrid useless | Self-RAG only skips for questions LLM knows. For domain-specific queries, retrieval still happens. | **No conflict** |
| Parent Document increases tokens → fights Compression | Complementary: Parent expands for completeness; Compression then optimizes. | **No conflict — synergistic** |
| CRAG rejects → triggers re-retrieval → increases latency | By design. CRAG's correction loop is a feature. Set max iterations. | **No conflict — configurable** |
| HyDE hypothetical answer → might bias retrieval | CRAG and Reranking validate against ORIGINAL query, neutralizing bias. | **No conflict — mitigated** |
| Multi Query too many results → overwhelms Reranking | Reranking scales to hundreds. Can deduplicate before reranking. | **No conflict — scalable** |

**Conclusion:** There are **genuinely zero conflicts** in this set. Every concept can be combined with every other. Constraints are only latency, cost, and complexity budgets.

---

## 4. How Every Combination Works

### 4.1 Detailed Interaction Explanations

#### Reranking × Everything

| With | Interaction | Result |
|------|-------------|--------|
| Hybrid Search | Reranker receives candidates from both vector and keyword arms; compares semantic relevance vs keyword density | Eliminates false positives from either method |
| Chunking | Better chunking → more coherent chunks → better reranker inputs | Reranking quality upper-bounded by chunk quality |
| Multi Query | Merges results from N queries; reranker deduplicates and creates unified ranking | Prevents query redundancy from inflating results |
| Parent Document | Reranker scores small chunks; top chunks trigger parent expansion | Precision at chunk level, completeness at doc level |
| Compression | Reranker ensures only quality chunks enter compression | Two-stage quality filter |
| HyDE | Reranker validates docs matching hypothetical answer ALSO match original query | Prevents HyDE hallucination from polluting results |
| Self-RAG | When Self-RAG says "YES" (retrieve), Reranking improves what gets retrieved | Conditional activation — no waste |
| CRAG | CRAG evaluates quality; if bad, Reranking reapplied after corrective loop | Double validation layer |

#### Hybrid Search × Everything

| With | Interaction | Result |
|------|-------------|--------|
| Chunking | Structure-aware chunking preserves headings/keywords BM25 catches; semantic chunking improves vector search | Chunking strategy directly impacts Hybrid Search |
| Multi Query | Each generated query runs through BOTH vector and keyword search | N queries x 2 methods = maximum surface area |
| Parent Document | Keyword search excels at finding exact section references; parent expansion provides full context | Best for document navigation |
| Compression | Hybrid may return more total candidates; compression handles volume | Volume management |
| HyDE | HyDE hypothetical answer provides rich text for BOTH embedding generation AND keyword extraction | Hypothetical answer feeds both arms optimally |
| Self-RAG | Self-RAG decides whether to search; Hybrid decides how to search | Orthogonal but compatible |
| CRAG | CRAG compares vector vs keyword result quality; detects when one arm fails | Quality monitoring per search arm |

#### Chunking × Everything

| With | Interaction | Result |
|------|-------------|--------|
| Multi Query | Semantic chunking ensures different phrasings hit same conceptual chunks | Consistent retrieval across query variations |
| Parent Document | Parent Document Chunking (Strategy #4) IS Parent Document Retrieval at chunking layer | Native integration |
| Compression | Smaller semantic chunks need less aggressive compression | Reduced information loss |
| HyDE | Semantic chunks align with topic clusters HyDE maps to | Better semantic alignment |
| Self-RAG | Self-RAG is query-agnostic; chunking is data-agnostic | Independent improvements |
| CRAG | CRAG's evaluator works better with coherent, complete chunks | Better evaluation inputs |

#### Multi Query × Everything

| With | Interaction | Result |
|------|-------------|--------|
| Parent Document | Multiple queries find chunks from different regions; parent ensures each brings full context | Multi-faceted questions get multi-faceted answers |
| Compression | Multi Query inflates pool; compression deflates it | Volume equilibrium |
| HyDE | Can generate multiple hypothetical answers OR use HyDE as one of multiple queries | Dual query enrichment |
| Self-RAG | Self-RAG gate determines IF Multi Query runs | Cost savings for known questions |
| CRAG | CRAG evaluates each query's result set; identifies bad angles | Query-quality-aware retrieval |

#### Parent Document × Everything

| With | Interaction | Result |
|------|-------------|--------|
| Compression | Parent expands chunks (more tokens); compression trims expanded parents | Perfect expansion-contraction balance |
| HyDE | HyDE finds right semantic neighborhood; parent ensures complete context | Semantic precision + contextual completeness |
| Self-RAG | Self-RAG decides whether to retrieve; parent determines what to return | Independent stages, no conflict |
| CRAG | CRAG evaluates PARENT documents for quality | Validates completeness |

#### Context Compression × Everything

| With | Interaction | Result |
|------|-------------|--------|
| HyDE | HyDE improves retrieval upstream; compression handles volume downstream | Sequential improvement |
| Self-RAG | When Self-RAG triggers retrieval, compression optimizes results | Conditional activation |
| CRAG | CRAG filters BEFORE compression | Quality gate before optimization |

#### HyDE × Everything

| With | Interaction | Result |
|------|-------------|--------|
| Self-RAG | Self-RAG can use query vagueness as signal: vague → activate HyDE; clear → skip | Adaptive HyDE activation |
| CRAG | CRAG validates HyDE-retrieved docs answer ORIGINAL question | Safety check against hypothetical bias |

#### Self-RAG × CRAG

| Interaction | Result |
|-------------|--------|
| Self-RAG decides IF to retrieve; CRAG evaluates WHAT was retrieved | Intelligence + validation. Perfect complementary pair. |

---

### 4.2 The Complete Combination Catalog

With 9 concepts, there are **511 non-empty combinations**. Below is the categorized catalog:

#### Category 1: Single-Concept Deployments (9)
Use when you have a specific, isolated problem:

| Concept | Use When | Expected Impact |
|---------|----------|-----------------|
| Reranking | Retrieval returns relevant docs but not in right order | +15-25% answer accuracy |
| Hybrid Search | Missing exact matches OR missing semantic matches | +20-30% recall |
| Chunking | Answers are incomplete or chunks cut mid-thought | +10-20% relevance |
| Multi Query | Same question phrased differently in docs vs queries | +25-35% recall |
| Parent Document | Retrieved chunks lack context (pronouns, references) | +15-20% completeness |
| Compression | Hitting token limits or "lost in the middle" | -50% tokens, +10% focus |
| HyDE | Users ask vague, short, or ambiguous questions | +20-30% recall for vague queries |
| Self-RAG | Many questions don't need retrieval (simple facts) | -30-50% retrieval costs |
| CRAG | LLM gives confident wrong answers from bad context | -40-60% hallucinations |

---

#### Category 2: Pairs (36)
Use when you have TWO related problems. See Section 2.2 for full details.

**Top 5 Most Impactful Pairs:**
1. **Reranking + Hybrid Search** — The production standard
2. **CRAG + Self-RAG** — Intelligence + validation
3. **Multi Query + HyDE** — Maximum recall for vague queries
4. **Parent Document + Context Compression** — Complete yet compact context
5. **Reranking + CRAG** — Double-layer quality assurance

---

#### Category 3: Triples (84)
Use when building a robust pipeline. See Section 2.3 for archetypes.

**Top 5 Most Impactful Triples:**
1. **Hybrid Search + Reranking + CRAG** — The production gold standard
2. **Self-RAG + CRAG + Multi Query** — Adaptive intelligence
3. **HyDE + Hybrid Search + Multi Query** — Maximum recall
4. **Reranking + Parent Document + CRAG** — Precision fortress
5. **Self-RAG + Reranking + Compression** — Efficiency master

---

#### Category 4: Quadruples (126)
Use for enterprise-grade systems. See Section 2.4 for patterns.

**Top 3 Patterns:**
1. **Hybrid + Reranking + CRAG + Compression** — Production Grade
2. **Self-RAG + Hybrid + Reranking + Compression** — Speed Demon
3. **Multi Query + HyDE + Hybrid + Parent Document** — Research Assistant

---

#### Category 5: Quintuples and Beyond (252+)
Use for mission-critical applications.

**The "Fort Knox" (6 concepts):** Hybrid + Multi Query + HyDE + Reranking + CRAG + Parent Document  
**The "All-In" (9 concepts):** All concepts combined

---

## 5. Implementation Roadmap

### 5.1 Beginner Path (Start Here)
**Components:** Chunking Strategies + Hybrid Search + Reranking

**Why:** These three concepts cover the fundamentals — how you split documents, how you search them, and how you order results. No LLM-based overhead. Fast to implement.

**Implementation:**
1. Choose chunking strategy (start with Fixed Size 512 tokens, 10% overlap)
2. Set up vector DB with both vector and BM25 indexes
3. Implement Reciprocal Rank Fusion (RRF) for merging
4. Add a lightweight cross-encoder reranker (e.g., BAAI/bge-reranker-base)

**Expected Outcome:** Solid baseline RAG system. ~70-80% of production quality.

---

### 5.2 Intermediate Path
**Add:** Parent Document Retrieval + Context Compression

**Why:** Parent Document fixes the "chunk out of context" problem. Compression fixes the "too much noise" problem.

**Implementation:**
1. Store parent_id with each chunk during indexing
2. After retrieval, map top-k chunks to parent sections
3. Implement extractive compression (keep top-N sentences by relevance)
4. Set compression target (e.g., reduce to 8 chunks or 2000 tokens)

**Expected Outcome:** Context quality improves dramatically. Answers become more complete and focused.

---

### 5.3 Advanced Path
**Add:** Multi Query + HyDE + Self-RAG

**Why:** These three make your system "smarter" — better at understanding vague questions, better at finding hidden info, better at avoiding unnecessary work.

**Implementation:**
1. Build Multi Query generator (LLM prompt that creates 3-5 diverse phrasings)
2. Build HyDE module (LLM generates hypothetical answer, embed it instead of query)
3. Build Self-RAG classifier (LLM decides YES/NO for retrieval need)
4. Add conditional logic: vague queries → HyDE; all queries → Multi Query; simple queries → Self-RAG bypass

**Expected Outcome:** System adapts to query type. Recall and efficiency both improve.

---

### 5.4 Expert Path
**Add:** CRAG

**Why:** CRAG is the final safety net. It catches everything else that might go wrong.

**Implementation:**
1. Build evaluator (LLM or cross-encoder scores retrieved docs on: relevance, completeness, consistency)
2. Set quality thresholds
3. Build corrective actions: refine query, try alternative search, filter noise
4. Set max iteration limit (e.g., 2 correction loops max)
5. Log all CRAG decisions for observability

**Expected Outcome:** Hallucinations and wrong answers drop significantly. System becomes self-correcting.

---

### 5.5 Decision Framework

```
                    START
                      │
                      ▼
        ┌─────────────────────────┐
        │  Is retrieval quality   │
        │  the main problem?      │
        └─────────────────────────┘
           │              │
          YES             NO
           │              │
           ▼              ▼
    ┌────────────┐  ┌─────────────────┐
    │ Add Hybrid │  │ Is context too  │
    │ + Rerank   │  │ noisy / too big?│
    └────────────┘  └─────────────────┘
       │                │         │
       │               YES        NO
       │                │         │
       │                ▼         ▼
       │         ┌──────────┐ ┌──────────────┐
       │         │ Add Parent│ │ Is recall    │
       │         │ + Compress│ │ too low?     │
       │         └──────────┘ └──────────────┘
       │            │            │        │
       │            │           YES       NO
       │            │            │        │
       │            │            ▼        ▼
       │            │    ┌──────────┐ ┌─────────────┐
       │            │    │ Add Multi│ │ Is cost/    │
       │            │    │ Query +  │ │ latency     │
       │            │    │ HyDE     │ │ too high?   │
       │            │    └──────────┘ └─────────────┘
       │            │       │            │       │
       │            │       │           YES      NO
       │            │       │            │       │
       │            │       │            ▼       ▼
       │            │       │    ┌──────────┐ ┌──────────┐
       │            │       │    │ Add Self │ │ Add CRAG │
       │            │       │    │ -RAG     │ │ (safety) │
       │            │       │    └──────────┘ └──────────┘
       │            │       │
       └────────────┴───────┘
                      │
                      ▼
              ┌──────────────┐
              │  PRODUCTION  │
              │    READY     │
              └──────────────┘
```

---

## Summary

| Concept | Stage | Core Value | Best Combined With |
|---------|-------|-----------|-------------------|
| **Reranking** | Post-Retrieval | Precision | Hybrid Search, CRAG, Parent Document |
| **Hybrid Search** | Retrieval | Recall | Reranking, Multi Query, CRAG |
| **Chunking** | Pre-Processing | Foundation | Parent Document, Semantic strategies |
| **Multi Query** | Pre-Retrieval | Recall | Hybrid Search, Reranking, CRAG |
| **Parent Document** | Post-Retrieval | Completeness | Reranking, Compression, CRAG |
| **Compression** | Post-Retrieval | Efficiency | Parent Document, CRAG, Reranking |
| **HyDE** | Pre-Retrieval | Query quality | Hybrid Search, Reranking, CRAG |
| **Self-RAG** | Pre-Retrieval | Efficiency | CRAG, Reranking, Compression |
| **CRAG** | Post-Retrieval | Validation | Reranking, Hybrid Search, Self-RAG |

---

*Document generated from @darpan.decoded's "10 Advanced RAG Concepts" Instagram reel.*
*All 9 concepts are mutually compatible with zero conflicts. The art is in choosing the right combination for your latency, cost, and accuracy requirements.*
