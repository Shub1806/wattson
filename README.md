# Wattson

**How many joules did that LLM request cost?**

Wattson is a small Rust gateway that sits in front of an LLM inference server
(vLLM / SGLang) and attributes GPU energy and cost to each individual request,
using NVIDIA's NVML energy counters.

> Status: work in progress.

## Layout

| Path | What it is |
|------|------------|
| `src/bin/mock-server.rs` | Fake OpenAI-compatible server for development without a GPU |
| `src/main.rs` | The Wattson gateway (in progress) |

## Run the mock server

```bash
cargo run --bin mock-server
curl -N localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"max_tokens":5,"messages":[{"role":"user","content":"hi"}]}'
```

Tune it with `MOCK_PREFILL_MS`, `MOCK_TOKEN_DELAY_MS` and `MOCK_PORT`.
