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
| `src/attribution.rs` | Splits GPU energy fairly between concurrent requests |
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
- **energy (J) and joules per token**: the request's fair share of GPU
  energy (see below), plus the naive figure for comparison

### The simulated GPU

`SimulatedGpu` models a T4-class card (70 W limit) using the phase each
in-flight request is in:

| State | Power | Why |
|-------|-------|-----|
| idle | 18 W | baseline draw |
| any request prefilling | +40 W | prompt processing is compute-bound |
| N requests decoding | +30 W × (1 − 0.7ᴺ) | decode is memory-bound; batching adds little power per extra request |

### Fair attribution

The GPU reports a single number: total energy used so far. The naive way to
charge a request (counter at the end minus counter at the start) also charges
it for every other request running at the same time.

Instead, `attribution.rs` closes a time slice every 20 ms, and whenever a
request starts, changes phase or finishes. It splits the energy used in that
slice between the active requests, weighting prefill 4× decode. Energy used
with no requests running goes to an `idle` bucket. Every joule is counted
exactly once.

Results with the simulated GPU, 40-token requests:

| Scenario | Naive (per request) | Fair (per request) | J/token |
|----------|--------------------:|-------------------:|--------:|
| 1 request alone | 25.6 J | 25.6 J | 0.64 |
| 4 concurrent | 37.0 J | 9.3 J | 0.23 |

Naive attribution overcounted by 4× under concurrency. With fair attribution
the four requests add up to what the GPU actually used, and batching shows up
as a 2.8× drop in energy per token.

Check it yourself: `GET /energy` shows `ledger.attributed_j + ledger.idle_j`,
which tracks `total_energy_j`.

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
