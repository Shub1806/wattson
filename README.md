# Wattson

**How many joules did that LLM request cost?**

Wattson is a small Rust gateway that sits in front of an LLM inference server
(vLLM / SGLang) and attributes GPU energy and cost to each individual request,
using NVIDIA's NVML energy counters.

```
client  ->  wattson (:8080)  ->  vLLM / SGLang / mock server (:8000)
```

> Status: work in progress. Energy currently comes from a simulated GPU so the
> gateway can be developed on any machine; real NVML readings come next.

## Layout

| Path | What it is |
|------|------------|
| `src/main.rs` | Gateway entry point and routes |
| `src/proxy.rs` | Forwards requests upstream, streams responses back, measures them |
| `src/energy.rs` | `EnergySource` trait, phase tracking, and the simulated GPU |
| `src/stats.rs` | In-memory per-request records and the `/stats` summary |
| `src/bin/mock-server.rs` | Fake OpenAI-compatible server for development without a GPU |

## What gets measured

For each request the gateway records:

- **completion tokens**: from the server's `usage` report when available
  (the gateway sets `stream_options.include_usage`), otherwise counted from
  the stream
- **time to first token (TTFT)**: roughly the *prefill* phase
- **decode throughput**: tokens per second after the first token
- **total latency**, and whether the client stayed until the end
- **energy (J) and joules per token**: GPU energy counter at the end minus
  at the start

### The simulated GPU

`SimulatedGpu` models a T4-class card (70 W limit) using the phase each
in-flight request is in:

| State | Power | Why |
|-------|-------|-----|
| idle | 18 W | baseline draw |
| any request prefilling | +40 W | prompt processing is compute-bound |
| N requests decoding | +30 W × (1 − 0.7ᴺ) | decode is memory-bound; batching adds little power per extra request |

### Known limitation: naive attribution

Right now each request is charged *all* GPU energy used while it ran. With
four concurrent requests, each was charged ~37 J, about 147 J in total, while
the GPU actually used ~37 J in that window. Splitting energy fairly between
concurrent requests is the next step.

## Try it

Terminal 1, the fake inference server:
```bash
cargo run --bin mock-server
```

Terminal 2, the gateway:
```bash
cargo run --bin wattson
```

Terminal 3, send a request through the gateway, then look at the stats:
```bash
curl -N localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"max_tokens":20,"messages":[{"role":"user","content":"hi"}]}'

curl localhost:8080/stats
curl localhost:8080/energy
```

## Configuration

| Variable | Default | |
|----------|---------|--|
| `WATTSON_PORT` | `8080` | Gateway port |
| `WATTSON_UPSTREAM` | `http://127.0.0.1:8000` | Inference server base URL |
| `WATTSON_ENERGY` | `sim` | Energy source (`sim`; `nvml` coming) |
| `MOCK_PREFILL_MS` | `50` | Mock server delay before the first token |
| `MOCK_TOKEN_DELAY_MS` | `20` | Mock server delay per generated token |
| `MOCK_PORT` | `8000` | Mock server port |
