//! A fake OpenAI-compatible LLM server.
//!
//! It pretends to be vLLM so the gateway can be developed on a machine
//! with no NVIDIA GPU (e.g. a Mac). It mimics the two phases of real
//! inference with simple delays:
//!   - prefill: one wait before the first token (MOCK_PREFILL_MS)
//!   - decode:  one wait per generated token     (MOCK_TOKEN_DELAY_MS)
//!
//! Endpoints:
//!   GET  /health
//!   POST /v1/chat/completions   (supports "stream": true via SSE)

use std::{convert::Infallible, env, net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::sleep;

const WORDS: &[&str] = &[
    "the", "gpu", "moves", "bytes", "from", "memory", "to", "compute", "units", "and",
    "decode", "is", "usually", "bandwidth", "bound", "while", "prefill", "is", "compute", "bound",
];

#[derive(Clone)]
struct Config {
    prefill: Duration,
    token_delay: Duration,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    #[serde(default)]
    messages: Vec<Value>,
}

fn env_ms(key: &str, default: u64) -> Duration {
    let ms = env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    Duration::from_millis(ms)
}

/// Rough prompt size: ~4 characters per token, like most BPE tokenizers.
fn estimate_prompt_tokens(messages: &[Value]) -> usize {
    let chars: usize = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .map(str::len)
        .sum();
    (chars / 4).max(1)
}

fn chunk(model: &str, content: Option<String>, finish: Option<&str>) -> Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": match content { Some(c) => json!({ "content": c }), None => json!({}) },
            "finish_reason": finish,
        }]
    })
}

async fn health() -> &'static str {
    "ok"
}

async fn chat(State(cfg): State<Arc<Config>>, Json(req): Json<ChatRequest>) -> Response {
    let n = req.max_tokens.unwrap_or(64).min(4096);
    let model = req.model.unwrap_or_else(|| "mock-model".to_string());
    let prompt_tokens = estimate_prompt_tokens(&req.messages);

    if !req.stream {
        sleep(cfg.prefill + cfg.token_delay * n as u32).await;
        let text: Vec<&str> = (0..n).map(|i| WORDS[i % WORDS.len()]).collect();
        return Json(json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text.join(" ") },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n,
                "total_tokens": prompt_tokens + n
            }
        }))
        .into_response();
    }

    // Streaming: one SSE event per token, then a finish chunk, then [DONE].
    let events = stream::unfold(0usize, move |i| {
        let cfg = cfg.clone();
        let model = model.clone();
        async move {
            let data = if i < n {
                sleep(if i == 0 { cfg.prefill } else { cfg.token_delay }).await;
                chunk(&model, Some(format!(" {}", WORDS[i % WORDS.len()])), None).to_string()
            } else if i == n {
                chunk(&model, None, Some("stop")).to_string()
            } else if i == n + 1 {
                "[DONE]".to_string()
            } else {
                return None;
            };
            Some((Ok::<_, Infallible>(Event::default().data(data)), i + 1))
        }
    });
    Sse::new(events).into_response()
}

#[tokio::main]
async fn main() {
    let cfg = Arc::new(Config {
        prefill: env_ms("MOCK_PREFILL_MS", 50),
        token_delay: env_ms("MOCK_TOKEN_DELAY_MS", 20),
    });
    let port: u16 = env::var("MOCK_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8000);

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat))
        .with_state(cfg);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("mock LLM server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
