# Designing Long-Term Memory for Conversational Agents

How should an agent retain useful information when conversation history exceeds its context window? This project implements six memory strategies and examines their tradeoffs in information preservation, retrieval, representation, and computational cost. The goal is to connect implementation failures to practical memory design decisions.

The [notebook](1_Long_Memory.ipynb) contains the implementations and study summary. The quantitative investigation focuses on History RAG and Graph RAG; it is not a benchmark ranking of all six methods.

## Design Progression

| Strategy / entry point | Design and tradeoff |
|---|---|
| **No Memory** — `chat_no_memory()` | Answer without history: minimal memory overhead, but earlier personal facts are unavailable. |
| **Chat History** — `chat_with_history()` | Keep recent messages within a token budget: preserves wording, but older facts disappear as the window fills. |
| **Summarization** — `chat_with_summary()` | Compress older dialogue while retaining recent messages: extends continuity, but adds LLM calls and can lose precise details. |
| **History RAG** — `chat_with_history_rag()` | Retrieve dated, speaker-labelled chunks using dense search, BM25, and reranking: scales beyond the context window, but memory remains raw text without explicit consolidation or forgetting. |
| **Structured Memory** — `chat_with_structured_memory()` | Extract `Episode` records into SQLite and expose a search tool: makes events inspectable, but schema omissions and per-message extraction cost become concerns. |
| **Graph RAG** — `chat_with_graph_rag()` | Retrieve connected entities and relations with supporting dialogue: enables relationship-based access, but extraction is lossy and indexing is expensive. |

The conceptual study distinguishes working, semantic, episodic, and procedural memory. The structured prototype implements episodic records (`who`, `action`, `whom`, `when`, `where`), not all four types.

This progression shifts the question from “how much history fits?” to “what information survives storage and reaches the model?” More structure improves some access patterns while introducing new opportunities to lose evidence.

## Implementation Findings and Changes

Graph RAG missed answers even when relevant information existed in the history. Investigation separated representation, retrieval, interpretation, and evaluation failures.

| Location | Failure and repair |
|---|---|
| `retrieve_graph_evidence()` | A **two-excerpt-per-source limit** discarded relevant dialogue inside retrieved sources. Rerank excerpts globally within a bounded evidence budget instead. |
| `retrieve_graph_evidence()` | Relations omitted facts, and mid-turn chunks lost speaker/date context. Supplement graph paths with **dense source retrieval and fine-grained BM25**, recover actual headers from preceding chunks, and present dialogue before relation hints. |
| `build_history_graph()` | The embedding adapter's asynchronous path split batches into individual calls. Set **`use_async=False` and `embed_batch_size=32`**. Embeddings matched within numerical tolerance; overall pipeline speedup was not established. |
| `grade_answer()` | Forced JSON Schema output sometimes contradicted the judge's reasoning. Request ordinary JSON, validate it with **`Grade.model_validate_json()`**, and stop when judge controls fail. Independently review final predictions because semantic grading errors remain. |

The answer model remains `qwen3:8b`; relation extraction uses `qwen3:4b-instruct`. The repairs preserve the original shared answer prompt and recover evidence without requiring a larger model.

## Evaluation

The structural audit covered all **500 LongMemEval-S-cleaned examples**. Sessions were not always chronological, turn-level `has_answer` flags missed some gold sessions, and some same-day evidence followed the question's clock time. These findings ruled out blindly filtering history by annotations or a strict timestamp cutoff. Reference answers and question types were used for evaluation, not retrieval or answer generation.

Accuracy comparison used **28 examples**, balanced across seven effective types including abstention: 14 for development and 14 for separate validation, selected with seed 42. Two independent model reviews, blinded to method names, agreed on all primary judgments.

| Split | History RAG | Fixed Graph RAG |
|---|---:|---:|
| Development | 12/14 | 13/14 |
| Validation | 6/14 | 7/14 |
| Overall | 18/28 (64.3%) | 20/28 (71.4%) |

This sample supports the repairs, not statistical superiority or a full-benchmark claim. Retrieval context budgets differ. Cache reuse and concurrent experiments prevent a reliable overall latency comparison.

A fresh-kernel run on **2026-11-05** completed all 15 code cells without errors and passed 6/6 judge controls. The two regression examples scored **1/2**:

| Demonstration case | Expected | Observed |
|---|---|---|
| Viewing recommendation | Netflix stand-up with strong storytelling | An Egypt documentary; incorrect despite relevant retrieved evidence |
| Age when receiving a necklace | 18 | 18; correct |

The [saved predictions](evaluation_results.csv) and [execution record](experiments/graph_rag_audit_20261104/report/notebook_execution.json) document this separate demonstration. Its 325.6-second runtime includes setup, baseline examples, and judge controls; it is not a method-level latency benchmark or a rerun of the 28-question comparison.

## Conclusions and Next Steps

- **Preserve source evidence:** summaries, schemas, and graphs can improve access while omitting details needed for later questions.
- **Diagnose retrieval and reasoning separately:** the eight remaining Graph RAG errors comprise three missing-evidence cases, four interpretation failures despite available evidence, and one mixed case.
- **Validate the evaluator:** passing judge controls does not establish reliable semantic grading on real examples.
- **Extend the study:** evaluate all 500 examples with equal context budgets, measure cold indexing and queries separately, and investigate consolidation and associative retrieval across episodic and semantic memory.

## Code and Evidence

Start with [1_Long_Memory.ipynb](1_Long_Memory.ipynb), using the local Ollama models and dependencies in [requirements.txt](requirements.txt). Its default evaluation selects **two Graph RAG examples**. This small execution demonstrates the complete pipeline; the 28-question comparison above is retained from the earlier experiments.

See the [final comparison](experiments/graph_rag_audit_20261104/report/comparison.csv), [validation summary](experiments/graph_rag_audit_20261104/report/summary.json), and [failure diagnosis](experiments/graph_rag_audit_20261104/report/failure_diagnosis.json). The [reproduction guide](experiments/README.md) provides a fresh-kernel notebook runner and an optional 28-question runner, both using the current notebook's unchanged logic. Final reports are retained; exploratory variants, duplicate snapshots, and graph caches are excluded from the repository. Historical notebook hashes identify the evaluated implementation before documentation and cleanup.
