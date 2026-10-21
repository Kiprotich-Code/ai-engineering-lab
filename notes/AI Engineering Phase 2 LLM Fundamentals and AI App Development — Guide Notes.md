# AI Engineering Phase 2: LLM Fundamentals and AI App Development

Study guide for backend developers pivoting to AI engineering. Based on Phase 2 of the Dataquest AI Engineer Roadmap (2-3 months; LLM APIs, prompting, function calling, MCP, FastAPI, Docker), with practical depth added for people who already ship backend code.

## 1. Orientation

### What Phase 2 is really about

AI engineers build applications with pre-trained models rather than training them. Your job is to connect a model to real data and actions, make it reliable, evaluate it, and deploy it. Phase 2 gives you the core loop: call a model, shape its behavior with prompts, let it trigger tools, wrap it in an API, and ship it in a container.

### Where your backend skills already transfer

| Backend skill you have | Where it shows up in AI engineering |
| --- | --- |
| REST design, HTTP, JSON | Every LLM provider is an HTTP API; tool schemas are JSON Schema |
| Auth, API keys, secrets | Key management, per-user quotas, gateway auth |
| Rate limiting, retries, backoff | Provider 429s and 5xx errors are routine; you must handle them |
| Caching | Response caching, prompt caching, semantic caching later |
| Queues, async, streaming | Token streaming (SSE), async calls, background jobs |
| Observability, logging | Logging prompts, tokens, latency, and cost per request |
| Docker, CI/CD | Same, plus secrets and model-key handling |
| Testing | Needs a new approach: outputs are probabilistic (evals) |

### What is genuinely new for you

- Non-deterministic outputs: the same input can produce different outputs.
- Cost measured in tokens, not requests.
- Prompts as a new kind of "code" that needs versioning and testing.
- Failure modes like hallucination, prompt injection, and silent quality drops.
- Evaluation instead of pass/fail unit tests.

### Suggested pace for you

The roadmap estimates 3-5 months total for software engineers moving to AI engineering, at 10-15 hours per week. You can likely skim the Python and API basics and spend your time on LLM behavior, tool use, and evaluation habits. A realistic plan for this phase is 8-10 weeks (see section 9).

## 2. Module 1: How LLMs Work

### Why it matters

You do not need to derive transformer math. You need a working mental model so you can choose models, predict cost, and debug odd behavior.

### Core concepts

**Tokens.** Models read and write tokens, which are chunks of text (roughly 3/4 of an English word on average, but this varies by language and content). You pay per token, and limits are in tokens. Non-English text and code often use more tokens per word.

**Context window.** The maximum tokens (input plus output) the model can handle in one call. Everything you want the model to know must fit: system prompt, conversation history, retrieved documents, tool definitions, and the answer. Bigger windows cost more and can dilute attention on the relevant parts.

**Next-token prediction.** An LLM generates one token at a time, each conditioned on all previous tokens. It is a very capable pattern completer, not a database or a calculator. This explains both its fluency and its hallucinations.

**Sampling parameters.**

- `temperature`: higher means more varied output, lower means more deterministic. Use low values for extraction, classification, and tool use; higher for brainstorming and creative writing.
- `top_p`: samples from the smallest set of tokens whose probabilities sum to p. Usually adjust temperature or top\_p, not both.
- `max_tokens`: cap on output length. Also a cost and latency control.
- `stop` sequences: strings that end generation.

**Stateless calls.** The API remembers nothing. "Memory" in a chatbot means you resend the conversation history on every call.

**Training stages (high level).** Pre-training on huge text corpora, then instruction tuning and preference tuning (RLHF-style methods) to make the model follow instructions. The model's knowledge stops at its training cutoff, which is why you later add retrieval (RAG) and tools.

**Limitations to internalize.**

- Hallucination: confident but wrong output.
- Knowledge cutoff: no awareness of recent events unless you provide them.
- Weak at exact arithmetic and counting without tools.
- Sensitive to prompt wording and position of information in long contexts.
- Can be manipulated through prompt injection (text in the input that tries to override your instructions).

### Model families and choosing a model

Know the main families: GPT (OpenAI), Claude (Anthropic), Gemini (Google), Llama (Meta, open-weight), Mistral, DeepSeek. Model names and versions change quickly, so always check each provider's current model list and pricing page rather than memorizing names.

Selection criteria to build a habit around:

| Criterion | Question to ask |
| --- | --- |
| Quality | Does it pass your own test set for this task? |
| Latency | Is it fast enough for interactive use? Does it stream? |
| Cost | Price per input and output token at your expected volume |
| Context window | Does your worst-case input fit? |
| Capabilities | Tool use, structured output, vision, reasoning modes |
| Privacy and compliance | Data retention, region, whether the data may leave your infrastructure |
| Open vs hosted | Open-weight models (Llama, Mistral, DeepSeek) can be self-hosted at the cost of ops work |

A common pattern: use a small, cheap model for simple or high-volume steps and a stronger model for hard steps (routing).

### Hands-on exercises

1. Use a tokenizer tool (for example OpenAI's `tiktoken` library) to count tokens for the same paragraph in English, code, and another language. Note the differences.
2. Call one model at temperature 0 and temperature 1 five times each with the same prompt. Compare variance.
3. Deliberately overflow the context window and read the error. Then implement a function that trims old messages to stay under a token budget.
4. Ask a model about something after its cutoff and about a made-up fact, and observe how it fails.
5. Build a small spreadsheet comparing three models on price, context size, and speed for one task you care about.

### Self-check

- Why is a chat "stateless" at the API level, and what does that mean for cost as conversations grow?
- What is the difference between context window and max output tokens?
- When would you set temperature near 0?

Dataquest course: AI Chatbots: Harnessing LLMs (free).

## 3. Module 2: Prompt Engineering

### Why it matters

The gap between a demo and a production feature is often prompt quality plus the code around it. Treat prompting as a daily engineering skill, not a trick.

### The chat API shape

Both main providers use a list of role-tagged messages.

**OpenAI-style chat completions** (system message sits in the messages list):

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a concise support assistant."},
        {"role": "user", "content": "How do I reset my password?"},
    ],
    temperature=0.2,
)
print(resp.choices[0].message.content)
print(resp.usage)  # prompt and completion token counts
```

**Anthropic Messages API** (system prompt is a top-level parameter and `max_tokens` is required):

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
resp = client.messages.create(
    model="claude-sonnet-4-5",  # check docs for current model names
    max_tokens=500,
    system="You are a concise support assistant.",
    messages=[{"role": "user", "content": "How do I reset my password?"}],
)
print(resp.content[0].text)
print(resp.usage)
```

Note: model identifiers change often. Look up current names in each provider's docs before running examples. Providers also offer newer API styles (for example OpenAI's Responses API), so skim the current docs for the recommended interface.

### Roles

- `system`: standing instructions, persona, rules, output format.
- `user`: the human's input (and, in your apps, untrusted data).
- `assistant`: previous model replies; you can also pre-fill or include examples.

### Techniques that work

**1. Be explicit and specific.** State the task, audience, constraints, and output format. Vague prompts get vague results.

**2. Give structure to the prompt.** Separate instructions from data using clear delimiters (XML-style tags or fenced blocks), for example `<document>...</document>`. This also reduces prompt injection risk.

**3. Few-shot examples.** Show 2-5 input/output examples. Pick diverse, representative examples, including an edge case.

**4. Let the model reason on hard tasks.** Ask it to think step by step or use a provider's reasoning/thinking mode for multi-step problems. For simple tasks, skip it to save tokens and latency.

**5. Specify the output format.** Ask for JSON that matches a schema, or use the provider's structured-output feature. Always validate the result in code (see Pydantic in Module 3).

**6. Assign a role and audience.** "You are a senior support agent writing for a non-technical customer."

**7. Tell it what to do when unsure.** For example: "If the answer is not in the provided text, say you do not know." This reduces hallucination.

**8. Break big tasks into steps (prompt chaining).** Several small, focused calls often beat one giant prompt, and you can test each step.

**9. Iterate against test cases.** Keep a small set of representative inputs with expected qualities and re-run them after every prompt change.

### A reusable prompt template

```text
ROLE: You are <role> helping <audience>.
TASK: <one clear sentence>.
CONTEXT:
<context> ... </context>
RULES:
- <constraint 1>
- <constraint 2>
- If information is missing, say so instead of guessing.
OUTPUT FORMAT: <schema or example>
EXAMPLES:
<example> input -> output </example>
```

### Managing conversation context and token budgets

Since calls are stateless, the history grows each turn and so does cost and latency. Strategies:

- Sliding window: keep the last N turns.
- Summarization: replace older turns with a model-written summary.
- Selective memory: store key facts separately and inject only the relevant ones.
- Count tokens before sending, and reserve room for the reply.
- Use prompt caching (offered by major providers) when a large, stable prefix such as a long system prompt is reused.

### Streaming

For chat UIs, stream tokens as they are generated so users see output immediately. Both providers support streaming; in your own API you typically expose it with Server-Sent Events (SSE).

### Common failure modes

- Prompt too long and unfocused, so key instructions get ignored.
- Contradictory instructions.
- Relying on the model for facts instead of providing sources.
- Trusting free-text output without validation.
- Changing a prompt without re-running tests.

### Hands-on exercises

1. Write a classification prompt (for example, support ticket category) and improve it from zero-shot to few-shot. Measure accuracy on 20 labeled samples.
2. Build a function `trim_history(messages, max_tokens)` and test it.
3. Ask for JSON output, intentionally break it with a tricky input, and write code that detects and retries.
4. Try a basic prompt injection ("ignore previous instructions") against your own prompt, then harden it with delimiters and clear rules.
5. Store prompts in files with version numbers rather than inline strings.

Dataquest course: Prompting LLMs in Python.

## 4. Module 3: Function Calling, Tool Use, and MCP

### Why it matters

This is where an LLM stops being a text generator and becomes an application component. The model can request actions such as querying a database, calling an API, or running code. You stay in control: the model proposes a tool call, and your code executes it.

### Structured outputs with Pydantic

Use Pydantic models to define the shape you expect, and validate model output against it.

```python
from pydantic import BaseModel, ValidationError
from typing import Literal

class Ticket(BaseModel):
    category: Literal["billing", "bug", "feature", "other"]
    urgency: int  # 1-5
    summary: str

def parse_ticket(raw_json: str) -> Ticket | None:
    try:
        return Ticket.model_validate_json(raw_json)
    except ValidationError as e:
        # log the error, then retry the model call with the error message
        return None
```

Pattern: request structured output (via the provider feature or tool schema), validate, and on failure retry with the validation error included in the follow-up prompt. Cap retries.

### The tool-calling loop

Concept (same for all providers, with different field names):

1. You send the messages plus a list of tool definitions (name, description, JSON Schema for parameters).
2. The model replies either with text or with one or more tool call requests.
3. Your code runs the requested function(s).
4. You send the results back as tool-result messages.
5. The model continues; repeat until it returns a final text answer.

OpenAI-style sketch:

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up the status of an order by its ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}]

messages = [{"role": "user", "content": "Where is order A123?"}]

for _ in range(5):  # hard cap on loop iterations
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools
    )
    msg = resp.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        print(msg.content)
        break
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_order_status(**args)  # your own function
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
```

Anthropic differences to learn: tools use `input_schema`; the model returns `tool_use` content blocks with `stop_reason == "tool_use"`; you reply with a user message containing `tool_result` blocks referencing the `tool_use_id`.

### Designing good tools

- Clear names and descriptions: the model chooses tools from these, so write them like documentation for a new teammate.
- Narrow, well-typed parameters with enums where possible.
- Return concise, structured results (not giant payloads that waste context).
- Return helpful error messages the model can act on.
- Idempotent where possible; require confirmation for destructive or financial actions.
- Validate arguments in your code. Never trust model-generated arguments.
- Apply least privilege: a tool should only be able to do what it must, with the calling user's permissions.

### Safety rules for tool loops

- Hard cap on iterations and total tokens.
- Timeouts for each tool call.
- Log every tool call and result.
- Treat tool outputs (web pages, emails, documents) as untrusted: they can contain injected instructions.
- Human approval step for high-impact actions (refunds, deletes, sending emails).

### MCP (Model Context Protocol)

MCP is an open protocol that standardizes how AI applications connect to tools and data. Instead of writing custom integrations for every app and model, you build a tool server once and any MCP-compatible client can use it.

Core ideas:

- **Server**: exposes capabilities. Main primitives are tools (actions), resources (readable data), and prompts (reusable templates).
- **Client/host**: the AI application that connects to servers and passes their tools to the model.
- **Transports**: local servers commonly use stdio; remote servers use HTTP-based transports. Check the current spec for details because it evolves.
- **Why it matters**: reuse, a growing ecosystem of ready-made servers, and a clean separation between your AI app and your tool code.

Minimal server using the official Python SDK (verify API against current docs):

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("orders")

@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Look up the status of an order by its ID."""
    return lookup_order(order_id)  # your own function

if __name__ == "__main__":
    mcp.run()  # defaults to stdio
```

Mental model for a backend dev: MCP is to AI tools roughly what OpenAPI plus a standard client is to REST services, with the added benefit that models can discover and call tools directly.

### Hands-on exercises

1. Implement two tools (for example, `get_weather` and `search_orders`) and a tool loop with an iteration cap.
2. Make the model try to call a tool with bad arguments; handle it gracefully and feed the error back.
3. Write a Pydantic model for structured extraction from messy text (invoices, emails) and measure how often validation passes.
4. Build a small MCP server exposing 2-3 tools and connect it to an MCP-capable client to test it.
5. Add an approval gate to any tool that changes data.

Dataquest course: Tool Use with LLMs in Python.

## 5. Module 4: APIs for AI Applications

### Why it matters

AI engineering means integrating model providers and third-party services. You already know REST, so focus on the details that bite in AI workloads.

### What to master

**Query parameters and filtering.** Build requests cleanly with `requests` or `httpx`, passing params as dicts rather than string concatenation.

**Authentication.**

- API keys in headers (`Authorization: Bearer ...` or provider-specific headers).
- Keep keys in environment variables or a secrets manager. Never commit them. Use `.env` files locally and add them to `.gitignore`.
- OAuth flows appear when integrating user-owned data (calendar, email).

**Rate limits.**

- Providers limit requests per minute and tokens per minute.
- Handle HTTP 429 with exponential backoff and jitter; respect `Retry-After` headers.
- Add client-side throttling and a concurrency limit (for example, an `asyncio.Semaphore`).

**Pagination.** Cursor-based and offset-based. Write a generic generator that walks pages until exhausted.

**Errors and resilience.**

- Distinguish retryable errors (429, 5xx, timeouts) from non-retryable (400, 401, 403).
- Set explicit timeouts on every call.
- Use idempotency where supported.

**Async and concurrency.** Calls to LLMs are slow (seconds). Use `httpx.AsyncClient` or the providers' async clients so one worker can serve many requests.

**Streaming (SSE).** Understand how event streams work so you can consume provider streams and re-expose them to your own clients.

**Cost and usage tracking.** Every response includes token usage; log it with user, feature, and model so you can attribute cost.

### Retry helper example

```python
import random, time
import httpx

RETRYABLE = {429, 500, 502, 503, 504}

def request_with_retry(method, url, *, retries=5, **kwargs):
    for attempt in range(retries):
        try:
            r = httpx.request(method, url, timeout=30, **kwargs)
            if r.status_code not in RETRYABLE:
                r.raise_for_status()
                return r
        except httpx.TransportError:
            pass
        time.sleep(min(2 ** attempt, 30) + random.random())
    raise RuntimeError("Request failed after retries")
```

### Hands-on exercises

1. Write a client for a public paginated API that yields all items lazily.
2. Add retry/backoff and a concurrency limit, then fire 50 parallel requests and watch it behave.
3. Build a usage logger that records model, tokens in/out, latency, and estimated cost per call.
4. Store keys in env vars, run a secret scanner on your repo (for example, gitleaks), and confirm nothing leaks.

Dataquest course: APIs for AI Applications.

## 6. Module 5: Building and Deploying AI Apps (FastAPI and Docker)

### Why it matters

The gap between "works in a notebook" and "runs in production" is where many learners stall. As a backend developer this is your advantage: apply your existing service-building habits to an LLM workload.

### FastAPI for LLM services

What to learn:

- Path operations, request/response models with Pydantic, validation errors.
- Async endpoints and dependency injection (for clients, config, auth).
- Streaming responses with `StreamingResponse` (SSE).
- Background tasks for slow, non-interactive work.
- Settings management (environment variables with `pydantic-settings`).
- Middleware for request IDs, logging, and timing.
- Auth and per-user rate limiting.
- Auto-generated OpenAPI docs at `/docs`.

Minimal chat endpoint with streaming:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()
client = AsyncOpenAI()

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    async def gen():
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": req.message}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### Design points specific to AI services

- Keep prompts and model names in config, not scattered in code.
- Return consistent error shapes and map provider errors to your own.
- Set request timeouts and a maximum input size.
- Store conversation state in a database or cache (Redis, Postgres), not in process memory, so you can scale horizontally.
- Never expose provider API keys to the client.
- Add structured logs with request ID, model, token counts, latency, and cost.
- Add guardrails: input length limits, basic content checks, output validation.

### Multi-provider LLM gateway (milestone concept)

A gateway puts one stable interface in front of several providers so you can swap or combine models without changing application code.

Typical features:

- Unified request/response schema (you define it) with adapters for OpenAI, Anthropic, Gemini, and others.
- Provider selection by model name or routing rules (cost, latency, task type).
- Fallback on failure or rate limit (try provider B if A fails).
- Retries, timeouts, and circuit breakers.
- Usage and cost logging per user and per model.
- Optional response caching and rate limiting per API key.

Suggested structure:

```text
app/
  main.py            # FastAPI app and routes
  config.py          # settings from environment
  schemas.py         # unified request/response models
  providers/
    base.py          # abstract Provider interface
    openai_provider.py
    anthropic_provider.py
  router.py          # routing and fallback logic
  usage.py           # token and cost accounting
tests/
Dockerfile
docker-compose.yml
```

### Docker for AI apps

What to learn:

- Writing a `Dockerfile` for a Python service.
- Multi-stage builds to keep images small.
- Running as a non-root user.
- Health checks (`HEALTHCHECK` or an orchestrator probe against `/health`).
- Passing secrets and config through environment variables at runtime, never baking keys into images.
- `.dockerignore` to avoid copying `.env`, `.git`, and caches.
- Docker Compose for multi-service setups (API plus Redis or Postgres).

Example Dockerfile:

```dockerfile
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Example `docker-compose.yml`:

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [redis]
  redis:
    image: redis:7-alpine
```

### Production checklist

- [ ] Secrets only from environment or a secrets manager
- [ ] Timeouts, retries, and concurrency limits on outbound LLM calls
- [ ] Per-user rate limits and max input size
- [ ] Token and cost logging with request IDs
- [ ] Health endpoint and container health check
- [ ] Non-root container, small image, pinned dependencies
- [ ] State outside the process (database or cache)
- [ ] Output validation and loop caps for any tool use
- [ ] A small automated test set for prompt/behavior regressions
- [ ] README with setup, config, and run instructions

### Hands-on exercises

1. Build `/chat` with and without streaming; test both with `curl`.
2. Add a dependency that enforces an API key and a simple per-key rate limit.
3. Containerize it, run it with Compose and Redis, and move conversation history into Redis.
4. Kill the provider (use a wrong key) and confirm your error handling and health check behave sensibly.

Dataquest course: Building AI Apps with FastAPI.

## 7. Milestone Projects

The roadmap names three milestone outcomes for Phase 2: a Dynamic AI Chatbot, a Multi-Provider LLM Gateway, and a complete AI service deployed with FastAPI and Docker.

### Project A: Dynamic AI Chatbot

Goal: a chatbot with selectable personas, managed context, and tool use.

Minimum scope:

- Conversation history with a token budget (sliding window or summary).
- System prompt chosen per persona or use case.
- At least two tools (for example, weather lookup and a notes/orders lookup).
- Streaming output.
- Structured output validated with Pydantic for at least one feature.

Stretch: persist sessions, add a simple web or CLI UI, add basic prompt injection defenses.

Done when: you can hold a 20-turn conversation without hitting the context limit, and tool calls succeed and fail gracefully.

### Project B: Multi-Provider LLM Gateway

Goal: one API in front of at least two providers.

Minimum scope:

- Unified schema and two provider adapters.
- Model routing and automatic fallback.
- Retry/backoff on 429 and 5xx.
- Usage and cost logging.

Stretch: response caching, per-key rate limits, streaming passthrough, simple dashboard of cost by model.

Done when: with provider A disabled, requests are still served by provider B and the logs show the switch.

### Project C: Deployed AI Service (FastAPI and Docker)

Goal: a containerized AI API someone else can run with one command.

Minimum scope:

- FastAPI app with documented endpoints and a health check.
- Docker image (multi-stage or slim, non-root) and Compose file with a data store.
- Config through environment variables.
- README with architecture diagram, setup, and example requests.

Stretch: CI pipeline that builds the image and runs tests, deploy to a cloud host, add basic auth.

Done when: a fresh clone plus `docker compose up` yields a working service.

### Portfolio tips

Push each project to GitHub with a clear README, architecture sketch, sample outputs, and known limitations. The source roadmap advises showing working projects early rather than waiting to feel ready.

## 8. Testing and Evaluation Mindset (start now, deepen in Phase 4)

The source roadmap places formal LLM evaluation in Phase 4, but start the habit now:

- Keep a small golden set (20-50 inputs) per feature with expected properties.
- Check programmatic things: valid JSON, required fields present, correct tool chosen, length limits.
- For subjective quality, review samples by hand; later, add LLM-as-judge scoring.
- Re-run the set after every prompt or model change and record results.
- Mock the provider in unit tests for the plumbing (routing, retries, parsing) so those tests are deterministic.

## 9. Suggested 10-Week Plan

| Week | Focus | Output |
| --- | --- | --- |
| 1 | How LLMs work, tokens, parameters, model comparison | Token/cost notebook, model comparison sheet |
| 2 | Chat APIs (OpenAI and Anthropic), history and budgets | Basic CLI chatbot with trimming |
| 3 | Prompt engineering techniques, few-shot, structured output | Prompt test set with measured accuracy |
| 4 | Pydantic validation, tool calling loop | Two-tool agent loop with caps |
| 5 | MCP server and client basics | Small MCP server with 2-3 tools |
| 6 | API integration: auth, rate limits, pagination, retries | Resilient client library |
| 7 | FastAPI service with streaming and config | Dynamic AI Chatbot (Project A) |
| 8 | Provider adapters, routing, fallback, cost logging | Multi-Provider LLM Gateway (Project B) |
| 9 | Docker, Compose, health checks, non-root images | Containerized service |
| 10 | Polish, tests, README, deployment | Deployed AI Service (Project C) and portfolio write-ups |

Adjust based on your starting point. If you already know FastAPI and Docker, compress weeks 7-10 and add time for evaluation and prompt injection testing.

## 10. Final Self-Assessment

You are ready for Phase 3/4 when you can do all of these without a tutorial:

- [ ] Explain tokens, context windows, and temperature, and estimate the cost of a conversation.
- [ ] Choose between two models for a task and justify it with data.
- [ ] Write a prompt with clear structure and few-shot examples, and show it improves a test set.
- [ ] Get validated, structured output from a model and handle failures.
- [ ] Implement a safe tool-calling loop and explain how it differs across OpenAI and Anthropic.
- [ ] Describe what MCP is and build a simple server.
- [ ] Call third-party APIs with proper auth, retries, rate-limit handling, and pagination.
- [ ] Build a streaming FastAPI service backed by an LLM.
- [ ] Containerize and run it with Docker Compose, following the production checklist.
- [ ] Explain at least three ways your app could be attacked (prompt injection, key leakage, runaway tool loops) and how you mitigated them.

## 11. Resources

**Dataquest courses mapped to Phase 2** (from the roadmap)

- AI Chatbots: Harnessing LLMs (free)
- Prompting LLMs in Python
- Tool Use with LLMs in Python
- APIs for AI Applications
- Building AI Apps with FastAPI

**Documentation to bookmark**

- OpenAI API: https://platform.openai.com/docs
- Anthropic API: https://docs.anthropic.com/
- FastAPI: https://fastapi.tiangolo.com/
- Model Context Protocol: https://modelcontextprotocol.io
- Docker docs: https://docs.docker.com
- LangChain: https://docs.langchain.com/ and LlamaIndex: https://docs.llamaindex.ai/ (useful ahead of Phase 4)

**Source**

- Dataquest, "How to Become an AI Engineer in 2026 (A Complete Roadmap)": https://www.dataquest.io/blog/ai-engineer-roadmap/

Note: the roadmap itself says agents and LLM evaluation content is still being added to Dataquest's path. Model names, API versions, and the MCP spec change quickly, so verify code samples against current provider docs before relying on them.
