# The Lab: Applied AI Engineering

**Building, evaluating, and deploying reliable AI systems.**

This repository tracks an eight-week engineering roadmap focused on building practical AI systems, measuring their performance, understanding their failure modes, and documenting what it takes to run them reliably.

The objective is not to build the biggest models or the flashiest demos. It is to develop the engineering judgment required to build AI systems that solve real problems.

> A small system with reproducible evaluations, production-minded design, and an honest failure analysis is more valuable than a flashy demo with no evidence behind it.

## Engineering Principles

- **Measure before optimizing.** Establish baselines before changing prompts, models, retrieval strategies, or pipelines.
- **Test real failure modes.** Evaluate incorrect answers, malformed inputs, timeouts, retries, adversarial inputs, and degraded dependencies.
- **Design for reliability.** Use validation, explicit state transitions, idempotency, observability, and appropriate security controls.
- **Track the economics.** Measure latency, token usage, API costs, throughput, and resource requirements.
- **Make results reproducible.** Version datasets, evaluation criteria, configurations, and relevant dependencies.
- **Document what did not work.** Negative results and failed approaches are part of the engineering record.

## Project Roadmap

Five projects build on a shared foundation. Retrieval, structured outputs, tool execution, evaluation, and observability developed in earlier projects will be reused wherever appropriate.

### 01 — Swahili/English RAG Assistant

**Focus:** Retrieval quality, grounding, and multilingual AI.

Build a retrieval-augmented generation (RAG) system over a useful document collection, such as Kenyan tax guidance, county service procedures, or travel requirements.

**Core implementation**
- Document ingestion, parsing, normalization, and chunking.
- Embedding generation and vector search using PostgreSQL and pgvector.
- Hybrid retrieval combining keyword and vector search.
- Reranking and context selection.
- Answers grounded in retrieved sources, with citations.
- Evaluation of retrieval and answer quality.

**Stack:** Python, FastAPI, PostgreSQL, pgvector, and a hosted LLM API.

**Evidence of completion**
- A curated evaluation set of 50–100 questions.
- Retrieval hit rate and answer-quality measurements.
- Baseline-versus-improved retrieval comparisons.
- Citation correctness and unsupported-answer analysis.
- A deployed demo and a documented failure analysis.

### 02 — Payment and Booking Agent

**Focus:** Tool calling, transactional correctness, and safe agent design.

Build an agent that checks availability, creates bookings, initiates a payment request through the M-Pesa Daraja sandbox, and verifies the resulting payment status.

The model proposes actions; deterministic application code enforces business rules and controls consequential operations.

**Core implementation**
- Explicit agent and booking state transitions.
- Typed tool schemas and validated arguments.
- Availability checks and booking creation.
- Daraja sandbox integration and asynchronous payment callbacks.
- Idempotency, retry handling, and duplicate-event protection.
- User confirmation before consequential actions.
- Prompt-injection defenses and authorization checks.

**Stack:** Python, FastAPI, Pydantic, PostgreSQL, and M-Pesa Daraja sandbox.

**Evidence of completion**
- Tests for repeated tool calls, incorrect amounts, stale availability, and duplicate callbacks.
- Verification that untrusted instructions cannot bypass business rules.
- Tests for timeouts, retries, interrupted workflows, and payment failures.
- A deployed sandbox demo and a documented threat and failure analysis.

### 03 — LLM Evaluation and Observability Harness

**Focus:** Systematic evaluation, regression testing, and operational visibility.

Extract reusable evaluation and monitoring components from the first two projects into a small library that can be used across AI applications.

**Core implementation**
- Versioned test datasets and evaluation runs.
- Deterministic checks for schemas, citations, and expected behavior.
- Model-assisted judging where appropriate, with documented judge limitations.
- Regression tests integrated into CI.
- Tracking of latency, token usage, estimated cost, errors, and model versions.
- Comparable evaluation reports across configurations.

**Stack:** Python, Pydantic, pytest, GitHub Actions, and a lightweight reporting interface.

**Evidence of completion**
- Reproducible evaluation runs.
- Regression checks triggered by relevant changes.
- A report showing at least one change that improved a metric while degrading another.
- Analysis of judge disagreement and measurement limitations.
- Documented overhead and known limitations.

### 04 — Messy-Document Extraction Pipeline

**Focus:** Multimodal processing, structured outputs, and data quality.

Build a pipeline that transforms receipts, invoices, M-Pesa statements, or scanned forms into validated structured records.

The system should distinguish extracted values from uncertain values and route low-confidence cases for human review rather than silently accepting questionable results.

**Core implementation**
- PDF, image, and text ingestion.
- OCR or vision-based extraction where appropriate.
- Structured output using Pydantic schemas.
- Field-level validation and confidence indicators.
- Human review and correction workflow.
- Batch processing, caching, and controlled retries.
- Safe handling of sensitive financial and personal data.

**Stack:** Python, Pydantic, a suitable document or vision API, PostgreSQL, and background processing where needed.

**Evidence of completion**
- A labeled dataset with a documented train/test separation where applicable.
- Field-level precision, recall, or exact-match accuracy as appropriate.
- Performance on malformed, incomplete, and low-quality documents.
- Cost per 1,000 documents, latency, and throughput.
- Analysis of extraction errors and human-review requirements.

### 05 — Fine-Tuning a Small Open Model

**Focus:** Model adaptation, experimental design, and deployment trade-offs.

Choose a narrow task, such as classifying Swahili, English, and Sheng customer messages by intent.

Fine-tune a suitable small open model using parameter-efficient fine-tuning such as LoRA, then compare it against a prompted hosted model on the same held-out dataset.

**Core implementation**
- Dataset preparation and label-quality checks.
- Leakage-resistant train, validation, and test splits.
- A prompted-model baseline.
- LoRA fine-tuning using Hugging Face tooling.
- Evaluation across language variants and underrepresented classes.
- Inference benchmarking and, where practical, quantization.

**Stack:** Python, Hugging Face Transformers, PEFT, and PyTorch. Use an available GPU environment such as Colab or Kaggle when suitable.

**Evidence of completion**
- Held-out classification metrics, including per-class performance.
- Comparison of fine-tuned and prompted models.
- Latency, deployment requirements, and estimated cost per prediction.
- Error analysis and checks for overfitting and data leakage.
- A clear recommendation on whether fine-tuning was justified.

## Eight-Week Execution Plan

| Week | Primary focus | Expected milestone |
|---|---|---|
| 1 | RAG ingestion and retrieval | Working baseline |
| 2 | RAG evaluation and deployment | Evaluated, deployed RAG demo |
| 3 | Booking agent and tool calling | End-to-end sandbox workflow |
| 4 | Payment safety and failure handling | Tested agent demo |
| 5 | Evaluation harness | Reusable evaluation library and CI |
| 6 | Document extraction | Validated extraction pipeline |
| 7 | Fine-tuning experiment | Baseline and initial training results |
| 8 | Final evaluation and portfolio polish | Final reports, deployment checks, and documented results |

This is an ambitious schedule. The milestones describe intended scope, not completed work or guaranteed outcomes.

If time becomes constrained, prioritize three strong deliverables: the RAG assistant, the payment and booking agent, and the evaluation harness. Reduce the extraction and fine-tuning projects to bounded experiments rather than sacrificing testing and documentation across every project.

## Shared Engineering Standards

Every project should include, where applicable:

- A clear problem statement and explicit scope.
- An architecture overview and setup instructions.
- Pinned or constrained dependencies and reproducible configuration.
- Automated tests and CI checks.
- Environment-variable-based configuration; no committed secrets.
- Input validation, appropriate authorization, and failure handling.
- A live demo or reproducible local execution path.
- Evaluation results, limitations, and known failure modes.

Projects handling payments or sensitive documents must use sandbox environments or appropriately anonymized data. Never commit credentials, payment secrets, personal records, or private datasets.

## Required Project Report

Every completed project must include a concise engineering report covering:

1. **Problem:** What real need does the system address?
2. **Design:** Why were these architectural and model choices made?
3. **Evaluation:** What dataset, baseline, metrics, and test conditions were used?
4. **Results:** What actually happened, including negative results?
5. **Failures:** Where does the system break, and why?
6. **Economics:** What are the measured costs and performance characteristics?
7. **Next steps:** What should change, and what evidence supports that decision?

Results must be reported as observed. Estimates must be labeled as estimates, and unmeasured claims must not be presented as established results.

## Progress Tracking

Project status is tracked explicitly:

- [ ] Planned
- [ ] In development
- [ ] Evaluated
- [ ] Deployed
- [ ] Documented

A project is not complete merely because its demo works. Completion requires the relevant tests, evaluation evidence, limitations, and documentation to be published.

## Success Criteria

At the end of this roadmap, the goal is to demonstrate the ability to:

- Build grounded, multilingual AI applications.
- Integrate models with transactional systems safely.
- Design useful evaluations and detect regressions.
- Process unstructured documents into validated data.
- Run controlled model experiments and justify model choices.
- Deploy, observe, and explain AI systems beyond the happy path.

**The standard is evidence, not hype.**

Small systems. Measured outcomes. Reproducible experiments. Honest failures.
