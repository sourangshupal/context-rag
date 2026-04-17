# Prompt Caching — Complete Reference

> Scope: LLM-level KV-cache prompt caching only.
> This document does NOT cover application-layer response caching (SQLiteCache, GPTCache, Redis, etc.).

---

## Table of Contents

1. [What Is Prompt Caching?](#1-what-is-prompt-caching)
2. [How It Works Internally](#2-how-it-works-internally)
3. [The Golden Rule](#3-the-golden-rule)
4. [Explicit Caching](#4-explicit-caching)
   - 4.1 Anthropic
   - 4.2 Google Vertex AI / Gemini
   - 4.3 AWS Bedrock
5. [Implicit Caching](#5-implicit-caching)
   - 5.1 OpenAI
   - 5.2 DeepSeek
   - 5.3 Google Gemini 2.5 (auto)
6. [Explicit vs Implicit — Side-by-Side](#6-explicit-vs-implicit--side-by-side)
7. [Pricing Summary](#7-pricing-summary)
8. [Framework Support](#8-framework-support)
   - 8.1 Raw SDK (Anthropic)
   - 8.2 LangChain
   - 8.3 LlamaIndex
   - 8.4 LiteLLM
9. [Prompt Structure for Maximum Cache Hits](#9-prompt-structure-for-maximum-cache-hits)
10. [This Codebase — How It Uses Prompt Caching](#10-this-codebase--how-it-uses-prompt-caching)
11. [Common Mistakes](#11-common-mistakes)
12. [Quick Decision Tree](#12-quick-decision-tree)

---

## 1. What Is Prompt Caching?

When an LLM processes a prompt, the most computationally expensive step is building the **KV (Key-Value) attention cache** — the internal representation of every token in your prompt. This computation is repeated on every single API call, even if 90% of the prompt is identical across requests.

Prompt caching lets the provider **save that computed KV cache** and **reuse it** across requests that share the same prefix, skipping the expensive recomputation entirely.

**Result:**
- **Speed**: 2–10× faster Time-To-First-Token (TTFT) on cache hits
- **Cost**: 50–90% cheaper for the cached portion of the prompt

**What gets cached:** The static prefix of your prompt (system prompt, documents, instructions).

**What never gets cached:** The dynamic suffix (user query, current chunk, variable data).

---

## 2. How It Works Internally

```
Request 1 (cache MISS — cold start):
┌─────────────────────────────────────────┐
│  [System prompt]         ← 200 tokens   │
│  [Full document]         ← 2000 tokens  │  → LLM computes KV cache
│  [Chunk 1 question]      ← 50 tokens    │     → SAVES KV cache to memory
└─────────────────────────────────────────┘
Cost: full price for 2250 tokens (+ small write premium on Anthropic)
TTFT: slow (processes all 2250 tokens)

Request 2 (cache HIT — same prefix):
┌─────────────────────────────────────────┐
│  [System prompt]         ← 200 tokens   │
│  [Full document]         ← 2000 tokens  │  → LOADS saved KV cache
│  [Chunk 2 question]      ← 50 tokens    │     → only processes 50 new tokens
└─────────────────────────────────────────┘
Cost: 90% discount on 2200 cached tokens, full price on 50 new tokens
TTFT: fast (only processes 50 tokens)
```

**Cache lookup is exact prefix match.** Even one token difference in the cached portion = full cache miss.

---

## 3. The Golden Rule

> **Static first. Dynamic last. Always.**

```
CORRECT structure (maximises cache hits):
┌──────────────────────────────────────┐
│  System prompt (never changes)       │  ← Cache breakpoint here
│  Instructions (never changes)        │
│  Reference document (per-document)   │  ← Cache breakpoint here
├──────────────────────────────────────┤
│  User query (changes every request)  │  ← Never cached
│  Current chunk (changes per chunk)   │  ← Never cached
└──────────────────────────────────────┘

WRONG structure (kills cache hits):
┌──────────────────────────────────────┐
│  User query (changes every request)  │  ← Dynamic content at the START
│  System prompt                       │     breaks prefix matching entirely
│  Reference document                  │
└──────────────────────────────────────┘
```

---

## 4. Explicit Caching

You explicitly tell the model **where** to place the cache breakpoint using special markers in the API request. The provider guarantees caching up to that marker.

---

### 4.1 Anthropic — `cache_control`

**Supported models:** All Claude 3+ models (Haiku, Sonnet, Opus)

**Mechanism:** Add `"cache_control": {"type": "ephemeral"}` to any content block.

**TTL options:**
| TTL | Write cost | Use case |
|-----|-----------|----------|
| `"ephemeral"` (5 min) | 1.25× base input price | Batch processing in one session |
| `"ephemeral"` + `"ttl": "1h"` | 2.00× base input price | Long-running agents, multi-session |

**Cache read cost:** 0.10× base input price (90% discount)

**Maximum breakpoints per request:** 4

#### Basic example (single document, one breakpoint)

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=500,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant that summarises document chunks.",
            "cache_control": {"type": "ephemeral"}   # Cache system prompt
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{full_document_text}\n</document>",
                    "cache_control": {"type": "ephemeral"}   # Cache document
                },
                {
                    "type": "text",
                    "text": f"Summarise this chunk in one sentence:\n\n{chunk_text}"
                    # No cache_control — this changes per chunk
                }
            ]
        }
    ]
)

# Inspect cache usage
usage = response.usage
print(f"Input tokens:            {usage.input_tokens}")
print(f"Cache creation tokens:   {usage.cache_creation_input_tokens}")   # what was written to cache
print(f"Cache read tokens:       {usage.cache_read_input_tokens}")       # what was served from cache
print(f"Output tokens:           {usage.output_tokens}")
```

#### 1-hour TTL (for agents or long sessions)

```python
{
    "type": "text",
    "text": full_document_text,
    "cache_control": {
        "type": "ephemeral",
        "ttl": "1h"          # costs 2x write but survives much longer
    }
}
```

#### Multi-turn conversation with cached system context

```python
# Cache the system prompt + knowledge base, keep conversation dynamic
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1000,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT + "\n\n" + KNOWLEDGE_BASE,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=conversation_history   # grows dynamically, not cached
)
```

#### Via OpenRouter (what this codebase uses)

OpenRouter passes `cache_control` through to Anthropic transparently:

```python
from openai import OpenAI   # OpenRouter uses OpenAI-compatible SDK

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

response = client.chat.completions.create(
    model="anthropic/claude-haiku-4-5",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": full_document,
                    "cache_control": {"type": "ephemeral"}   # passed through to Anthropic
                },
                {
                    "type": "text",
                    "text": chunk_question
                }
            ]
        }
    ]
)

# Cache tokens appear in usage
print(response.usage.prompt_cache_write_tokens)
print(response.usage.prompt_cache_read_tokens)
```

---

### 4.2 Google Vertex AI / Gemini — `CachedContent` API

Gemini's explicit caching is a separate API object — fundamentally different from Anthropic's inline markers. You create a named cache upfront and reference it in subsequent requests.

**Supported models:** Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 1.5 Pro

**Minimum cache size:** 32,768 tokens (Gemini 1.5) / ~1,024 tokens (Gemini 2.5)

**Minimum TTL:** 1 hour

**Cache read discount:** ~75% off

```python
import datetime
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

# Step 1: Create the cache ONCE (runs once per document session)
cache = genai.caching.CachedContent.create(
    model="models/gemini-2.5-flash",
    display_name="my-rag-document",
    system_instruction="You extract context from document chunks.",
    contents=[
        genai.protos.Content(
            parts=[genai.protos.Part(text=full_document_text)],
            role="user"
        )
    ],
    ttl=datetime.timedelta(hours=1),
)

print(f"Cache created: {cache.name}")
print(f"Token count: {cache.usage_metadata.total_token_count}")

# Step 2: Reference the cache in each chunk request
model = genai.GenerativeModel.from_cached_content(cached_content=cache)

for chunk in chunks:
    response = model.generate_content(
        f"Summarise this chunk:\n\n{chunk}"
    )
    print(response.text)

# Step 3: Delete cache when done (to stop TTL billing)
cache.delete()
```

---

### 4.3 AWS Bedrock — `cachePoint`

**Supported models:** Claude models on Bedrock

**Syntax:** `"cachePoint": {"type": "default"}` (instead of Anthropic's `cache_control`)

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.invoke_model(
    modelId="anthropic.claude-haiku-4-5-20251001-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_document
                    },
                    {
                        "type": "cachePoint",          # Bedrock-specific syntax
                        "cachePoint": {"type": "default"}
                    },
                    {
                        "type": "text",
                        "text": chunk_question
                    }
                ]
            }
        ]
    })
)
```

---

## 5. Implicit Caching

No special markers. The provider's infrastructure automatically detects repeated prefixes across requests and serves them from cache. You write a completely normal API call.

---

### 5.1 OpenAI — Automatic Prefix Caching

**Supported models:** GPT-4o, GPT-4o Mini, o1, o3, GPT-5 series

**Minimum prefix length:** 1,024 tokens

**Cache read discount:** 50% off input tokens

**Cache lifetime:** ~5–10 minutes of inactivity

**Key difference from Anthropic:** OpenAI caches are potentially shared across users of the same organisation. This is why system prompts across your entire app benefit, not just a single session.

```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Completely normal API call — zero special syntax
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT          # static — will be cached after first call
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_document},      # static — cached
                {"type": "text", "text": current_chunk_q},    # dynamic — not cached
            ]
        }
    ]
)

# Cache stats are surfaced in usage automatically
cached = response.usage.prompt_tokens_details.cached_tokens
total  = response.usage.prompt_tokens
print(f"Cache hit rate: {cached / total * 100:.1f}%")
```

---

### 5.2 DeepSeek — Automatic Context Caching

**Supported models:** DeepSeek V3, DeepSeek R1, DeepSeek V3.2

**Minimum prefix length:** 64 tokens (most aggressive minimum in the industry)

**Cache read discount:** ~90% off (same as Anthropic, but automatic)

**Cache lifetime:** A few minutes of inactivity

```python
from openai import OpenAI   # DeepSeek uses OpenAI-compatible API

client = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
)

# Completely normal call — caching is automatic
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},    # auto-cached after first call
        {"role": "user",   "content": full_document},    # auto-cached after first call
        {"role": "user",   "content": chunk_question},   # dynamic — not cached
    ]
)

# Cache stats
print(f"Cache hit tokens:  {response.usage.prompt_cache_hit_tokens}")
print(f"Cache miss tokens: {response.usage.prompt_cache_miss_tokens}")
```

**Via OpenRouter (no code change needed):**

```python
# Same call, just change model name and base URL
response = client.chat.completions.create(
    model="deepseek/deepseek-chat-v3-2",   # via OpenRouter
    messages=[...]                          # automatic caching still applies
)
```

---

### 5.3 Google Gemini 2.5 — Implicit Automatic Caching

**Supported models:** Gemini 2.5 Flash, Gemini 2.5 Pro

**Behaviour:** Automatic, no configuration. Gemini 2.5 caches common prefixes across all requests to the same model automatically.

**Cache read discount:** ~75% off

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-flash")

# Normal call — implicit caching happens automatically
response = model.generate_content([
    full_document,   # will be cached after first call if long enough
    chunk_question   # dynamic
])

# Usage metadata
print(response.usage_metadata.cached_content_token_count)
```

---

## 6. Explicit vs Implicit — Side-by-Side

| Dimension | Explicit | Implicit |
|---|---|---|
| **Who controls it** | You (developer) | The provider |
| **Code changes required** | Yes — add markers | None |
| **Guarantee of caching** | ✅ Guaranteed if TTL valid | ❌ Best-effort, provider decides |
| **Cache hit visibility** | Detailed token breakdown | Varies (sometimes none) |
| **TTL control** | Yes — 5 min or 1 hour (Anthropic) | No — provider decides |
| **Multi-user sharing** | No — per-account only | OpenAI: yes; others: no |
| **Minimum size** | 0 tokens (Anthropic marks any block) | 64–1024 tokens depending on provider |
| **Write premium** | Yes (1.25× on Anthropic) | No — first call = normal price |
| **Discount on hit** | 90% (Anthropic), 75% (Vertex) | 90% (DeepSeek), 50% (OpenAI), 75% (Gemini) |
| **Best for** | Large documents, controlled batches | Shared system prompts, agents |
| **Providers** | Anthropic, Google Vertex, AWS Bedrock | OpenAI, DeepSeek, Gemini 2.5 |

---

## 7. Pricing Summary

### LLM pricing with caching (per 1M tokens, April 2026)

| Provider | Model | Normal input | Cache write | Cache read |
|---|---|---|---|---|
| Anthropic | Claude Haiku 4.5 | ~$0.80 | ~$1.00 (1.25×) | ~$0.08 (90% off) |
| Anthropic | Claude Sonnet 4 | ~$3.00 | ~$3.75 (1.25×) | ~$0.30 (90% off) |
| DeepSeek | V3.2 | $0.14 | $0.14 (no premium) | ~$0.014 (90% off) |
| OpenAI | GPT-4o Mini | $0.15 | $0.15 (no premium) | $0.075 (50% off) |
| Google | Gemini 2.5 Flash | $0.075 | $0.075 (no premium) | ~$0.019 (75% off) |

### Effective cost per 1,000 chunks (RAG contextual embedding)

Assuming: 2,000-token document + 100-token chunk × 1,000 chunks, 70% cache hit rate.

| Setup | Cost / 1000 chunks |
|---|---|
| Anthropic Haiku (explicit, no cache) | ~$2.40 |
| **Anthropic Haiku (explicit, with cache)** | **~$0.65** |
| DeepSeek V3.2 (implicit, with cache) | ~$0.08 |
| OpenAI GPT-4o Mini (implicit, with cache) | ~$0.26 |
| Gemini 2.5 Flash (implicit, with cache) | ~$0.05 |

---

## 8. Framework Support

### 8.1 Raw Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": document_text,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": user_query
            }
        ]
    }]
)

# Usage
print(response.usage.cache_creation_input_tokens)
print(response.usage.cache_read_input_tokens)
```

---

### 8.2 LangChain

> **Important distinction:** LangChain has TWO separate caching concepts. Do not confuse them.

**LangChain's own cache** = caches the final LLM *response* in a local database. Entirely different from prompt caching.

```python
# LangChain's own response cache (NOT prompt caching)
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
# Same question → returns stored answer, skips LLM entirely
```

**Provider-level prompt caching via LangChain:**

LangChain passes `cache_control` through to Anthropic correctly via `langchain-anthropic`:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-haiku-4-5")

# Pass cache_control in the content block
response = llm.invoke([
    HumanMessage(content=[
        {
            "type": "text",
            "text": document_text,
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            "text": user_query
        }
    ])
])
```

**LangChain with AWS Bedrock prompt caching:**

```python
from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="anthropic.claude-haiku-4-5",
    beta_use_converse_api=True,
)

# Bedrock uses cachePoint syntax
response = llm.invoke([
    HumanMessage(content=[
        {"type": "text", "text": document_text},
        {"type": "cachePoint", "cachePoint": {"type": "default"}},
        {"type": "text", "text": user_query}
    ])
])
```

---

### 8.3 LlamaIndex

LlamaIndex passes through provider-level caching transparently. Add `cache_control` to the message content and it reaches Anthropic correctly.

```python
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.core.llms import ChatMessage, MessageRole

llm = LlamaAnthropic(model="claude-haiku-4-5")

messages = [
    ChatMessage(
        role=MessageRole.USER,
        content=[
            {
                "type": "text",
                "text": document_text,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": user_query
            }
        ]
    )
]

response = llm.chat(messages)
```

LlamaIndex also has its own `IngestionCache` for storing embeddings (again — separate concept, not prompt caching):

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.kvstore import SimpleKVStore

pipeline = IngestionPipeline(
    transformations=[...],
    cache=IngestionCache(cache=SimpleKVStore(), collection="my_cache")
)
# Skips re-embedding nodes that were already embedded
```

---

### 8.4 LiteLLM (most unified)

LiteLLM provides the most consistent cross-provider caching interface. One library, all providers.

```python
import litellm

# Anthropic — explicit cache_control
response = litellm.completion(
    model="anthropic/claude-haiku-4-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": document_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user_query}
        ]
    }]
)

# DeepSeek — implicit, nothing special needed
response = litellm.completion(
    model="deepseek/deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": document_text + user_query}
    ]
)

# OpenAI — implicit, nothing special needed
response = litellm.completion(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": document_text + user_query}
    ]
)

# LiteLLM normalises cache usage stats across providers
print(litellm.utils.get_prompt_cache_usage(response))
```

---

## 9. Prompt Structure for Maximum Cache Hits

### For RAG / Contextual Embedding (batch document processing)

```python
# OPTIMAL structure — maximises cache reuse across chunks of one document
SYSTEM = "You generate brief context for document chunks."  # ← cache breakpoint 1

for chunk in document_chunks:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=[{
            "type": "text",
            "text": SYSTEM,
            "cache_control": {"type": "ephemeral"}        # ← cache breakpoint 1
        }],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>{full_document}</document>",
                    "cache_control": {"type": "ephemeral"}  # ← cache breakpoint 2
                    # Same document cached across ALL chunks in this batch
                },
                {
                    "type": "text",
                    "text": f"Generate context for:\n{chunk}"  # changes every call
                }
            ]
        }]
    )
```

### For Chat Agents (multi-turn, long system context)

```python
# Cache the system prompt + tool definitions (thousands of tokens)
# Keep conversation history dynamic (grows with each turn)
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1000,
    system=[{
        "type": "text",
        "text": SYSTEM_PROMPT + TOOL_DEFINITIONS + KNOWLEDGE_BASE,
        "cache_control": {"type": "ephemeral", "ttl": "1h"}  # 1-hour TTL for agents
    }],
    messages=conversation_history   # dynamic — not cached
)
```

### For Multi-Document RAG (different document per request)

```python
# Use TWO cache breakpoints: one for instructions, one per document
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=500,
    system=[{
        "type": "text",
        "text": RETRIEVAL_INSTRUCTIONS,
        "cache_control": {"type": "ephemeral"}   # breakpoint 1 — never changes
    }],
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": retrieved_document,          # changes per document
                "cache_control": {"type": "ephemeral"}   # breakpoint 2 — per-doc cache
            },
            {
                "type": "text",
                "text": user_question                # dynamic — never cached
            }
        ]
    }]
)
```

---

## 10. This Codebase — How It Uses Prompt Caching

**File:** `context_rag_advanced_part1.ipynb` — `ContextualVectorDB` class

**Provider:** Anthropic Claude Haiku via OpenRouter

**Type:** Explicit caching with `cache_control: ephemeral` (5-min TTL)

**What is cached:** The full source document (sent once per chunk, cached after the first chunk of each document)

**What is NOT cached:** The individual chunk text (changes per API call)

**Prompts used:**

```python
# Prompt 1: Document context — CACHED
DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

# Prompt 2: Chunk instruction — NOT cached (dynamic)
CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk
within the overall document for the purposes of improving
search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""
```

**API call structure (simplified):**

```python
response = openrouter_client.chat.completions.create(
    model=LLM_MODEL,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                "cache_control": {"type": "ephemeral"}   # ← CACHED
            },
            {
                "type": "text",
                "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
                # ← NOT cached
            }
        ]
    }]
)
```

**Token tracking (thread-safe):**

```python
self.token_counts = {
    "input":            0,   # normal uncached tokens
    "output":           0,   # response tokens
    "cache_read":       0,   # tokens served from cache (cheap)
    "cache_creation":   0,   # tokens written to cache (small premium)
}
```

**Savings pattern:**

```
Chunk 1 of document A:  cache MISS → writes document A to cache
Chunk 2 of document A:  cache HIT  → reads document A from cache (90% off)
Chunk 3 of document A:  cache HIT  → reads document A from cache (90% off)
...
Chunk N of document A:  cache HIT  → reads document A from cache (90% off)

Chunk 1 of document B:  cache MISS → overwrites cache with document B
Chunk 2 of document B:  cache HIT  → reads document B from cache (90% off)
...
```

**Typical savings:** 60–70% reduction in total LLM cost for the contextual embedding pass.

---

## 11. Common Mistakes

| Mistake | Why it kills caching | Fix |
|---|---|---|
| Dynamic content at the start of the prompt | Breaks prefix matching — cache never hits | Move all dynamic content to the END |
| Putting a timestamp or request ID in the cached portion | Every request has a different prefix | Remove variable data from cached blocks |
| Processing chunks of different documents interleaved | Cache gets overwritten on every new document | Process all chunks of one document together before moving to next |
| Setting TTL too short | Cache expires mid-batch | Use 5-min TTL only for small batches; use 1-hour for large ones |
| Forgetting the minimum token threshold | Caching short prompts does nothing | Ensure cached block is at least ~200 tokens (Anthropic) |
| Using implicit caching with Anthropic | Anthropic requires explicit markers — it is NOT automatic | Always include `cache_control` when using Claude |
| Using `cache_control` with DeepSeek/OpenAI | These providers ignore the field — no error, but you may assume it worked | Just omit `cache_control` for non-Anthropic providers |

---

## 12. Quick Decision Tree

```
Are you using Anthropic / Claude?
│
├── YES → Use EXPLICIT caching (cache_control: ephemeral)
│         • Add marker to system prompt and/or document block
│         • Process all chunks of one document before the next
│         • Check cache_read_input_tokens in response.usage
│
└── NO → Which provider?
         │
         ├── DeepSeek → IMPLICIT (automatic, no code needed)
         │              90% discount, works for any prefix >64 tokens
         │
         ├── OpenAI  → IMPLICIT (automatic, no code needed)
         │              50% discount, works for prefixes ≥1024 tokens
         │
         ├── Gemini 2.5 Flash/Pro → IMPLICIT (automatic) OR EXPLICIT (CachedContent API)
         │              Use implicit for short sessions
         │              Use explicit CachedContent for documents >32k tokens or 1h+ sessions
         │
         └── AWS Bedrock (Claude) → EXPLICIT with cachePoint syntax
                        Same as Anthropic but different field name
```

---

*Last updated: April 2026*
*Stack context: Python 3.12 · Anthropic SDK 0.76 · OpenRouter 0.9 · Claude Haiku 4.5*
