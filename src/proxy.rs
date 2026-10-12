//! Forwards OpenAI-style chat requests to the upstream inference server
//! (vLLM, SGLang, or the mock server) and measures each one.
//!
//! Streaming responses are passed through to the client unchanged, chunk by
//! chunk. While they pass, we watch the Server-Sent Events to count tokens
//! and time the first one.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::{
    attribution::{Ledger, Meter},
    energy::{Activity, EnergySource, Phase},
    stats::{RequestRecord, Stats},
};

#[derive(Clone)]
pub struct AppState {
    pub client: reqwest::Client,
    pub upstream: String,
    pub stats: Stats,
    pub next_id: Arc<AtomicU64>,
    pub energy: Arc<dyn EnergySource>,
    pub activity: Arc<Activity>,
    pub ledger: Arc<Ledger>,
}

/// Energy-counter reading taken when a request starts.
struct EnergyStart {
    source: Arc<dyn EnergySource>,
    start_mj: u64,
}

impl EnergyStart {
    fn new(source: Arc<dyn EnergySource>) -> Self {
        let start_mj = source.total_energy_mj();
        Self { source, start_mj }
    }

    /// Joules used by the *whole GPU* since this request started.
    ///
    /// This is the naive measurement: it also includes any other requests
    /// running at the same time. Kept alongside the fair share (see
    /// attribution.rs) so the two can be compared.
    fn joules_so_far(&self) -> f64 {
        self.source.total_energy_mj().saturating_sub(self.start_mj) as f64 / 1000.0
    }
}

pub async fn chat_completions(State(state): State<AppState>, Json(mut body): Json<Value>) -> Response {
    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let start = Instant::now();
    let energy = EnergyStart::new(state.energy.clone());
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // Ask the server to report exact token usage at the end of the stream.
    // vLLM and SGLang support this; servers that don't simply ignore it.
    if stream {
        body["stream_options"] = serde_json::json!({ "include_usage": true });
    }

    let url = format!("{}/v1/chat/completions", state.upstream);
    let upstream = match state.client.post(&url).json(&body).send().await {
        Ok(resp) => resp,
        Err(err) => {
            return (StatusCode::BAD_GATEWAY, format!("upstream unreachable: {err}\n")).into_response()
        }
    };

    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok())
        .unwrap_or(HeaderValue::from_static("application/json"));

    let response_body = if stream && status.is_success() {
        // Streaming: we see the first token, so we know when prefill ends.
        let meter = Meter::start(state.ledger.clone(), state.activity.clone(), id, Phase::Prefill);
        stream_and_measure(upstream, id, start, energy, meter, state.stats)
    } else {
        // Non-streaming: we can't see phases, so count it all as decode.
        let meter = Meter::start(state.ledger.clone(), state.activity.clone(), id, Phase::Decode);
        buffer_and_measure(upstream, id, start, energy, meter, stream, status, state.stats).await
    };

    let mut response = Response::new(response_body);
    *response.status_mut() = status;
    response.headers_mut().insert(header::CONTENT_TYPE, content_type);
    response
        .headers_mut()
        .insert("x-wattson-request-id", HeaderValue::from(id));
    response
}

/// Non-streaming: read the whole response, take token counts from `usage`.
async fn buffer_and_measure(
    upstream: reqwest::Response,
    id: u64,
    start: Instant,
    energy: EnergyStart,
    mut meter: Meter,
    stream: bool,
    status: StatusCode,
    stats: Stats,
) -> Body {
    let bytes = upstream.bytes().await.unwrap_or_default();
    if status.is_success() {
        let usage = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| v.get("usage").cloned());
        let get = |key: &str| usage.as_ref().and_then(|u| u.get(key)).and_then(Value::as_u64);
        let completion_tokens = get("completion_tokens").unwrap_or(0);
        let energy_j = meter.finish();
        stats.push(RequestRecord {
            id,
            stream,
            prompt_tokens: get("prompt_tokens"),
            completion_tokens,
            energy_j,
            naive_energy_j: energy.joules_so_far(),
            joules_per_token: per_token(energy_j, completion_tokens),
            ttft_ms: None,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            decode_tokens_per_s: None,
            completed: true,
        });
    }
    Body::from(bytes)
}

/// Streaming: pass each chunk straight through while a `Tracker` watches it.
fn stream_and_measure(
    upstream: reqwest::Response,
    id: u64,
    start: Instant,
    energy: EnergyStart,
    meter: Meter,
    stats: Stats,
) -> Body {
    let tracker = Arc::new(Mutex::new(Tracker::new(id, start, energy, meter, stats)));
    let chunks = upstream.bytes_stream().map(move |chunk| {
        if let Ok(bytes) = &chunk {
            tracker.lock().unwrap().observe(bytes);
        }
        chunk
    });
    // When the stream finishes (or the client disconnects), `chunks` is
    // dropped, which drops the last Arc<Tracker> and runs Tracker::drop.
    Body::from_stream(chunks)
}

/// Watches a Server-Sent Events stream and records the request when done.
struct Tracker {
    id: u64,
    start: Instant,
    energy: EnergyStart,
    meter: Meter,
    stats: Stats,
    /// Holds a partial line when an SSE event is split across network chunks.
    buffer: String,
    counted_tokens: u64,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    usage_prompt: Option<u64>,
    usage_completion: Option<u64>,
    done: bool,
}

impl Tracker {
    fn new(id: u64, start: Instant, energy: EnergyStart, meter: Meter, stats: Stats) -> Self {
        Self {
            id,
            start,
            energy,
            meter,
            stats,
            buffer: String::new(),
            counted_tokens: 0,
            first_token_at: None,
            last_token_at: None,
            usage_prompt: None,
            usage_completion: None,
            done: false,
        }
    }

    fn observe(&mut self, bytes: &Bytes) {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        while let Some(pos) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=pos).collect();
            if let Some(data) = line.trim().strip_prefix("data:") {
                self.handle_event(data.trim());
            }
        }
    }

    fn handle_event(&mut self, data: &str) {
        if data == "[DONE]" {
            self.done = true;
            return;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else { return };

        // Exact counts, if the server sent a usage chunk.
        if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
            self.usage_prompt = usage.get("prompt_tokens").and_then(Value::as_u64);
            self.usage_completion = usage.get("completion_tokens").and_then(Value::as_u64);
        }

        // Otherwise, count events that carry new text. (Usually one token
        // per event, but servers may batch several, hence the usage chunk.)
        let has_text = event["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty());
        if has_text {
            let now = Instant::now();
            if self.first_token_at.is_none() {
                // First token out: prefill is over, decode begins.
                self.first_token_at = Some(now);
                self.meter.enter(Phase::Decode);
            }
            self.last_token_at = Some(now);
            self.counted_tokens += 1;
        }
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        let tokens = self.usage_completion.unwrap_or(self.counted_tokens);
        let decode_tokens_per_s = match (self.first_token_at, self.last_token_at) {
            (Some(first), Some(last)) if tokens > 1 && last > first => {
                Some((tokens - 1) as f64 / (last - first).as_secs_f64())
            }
            _ => None,
        };
        let energy_j = self.meter.finish();
        self.stats.push(RequestRecord {
            id: self.id,
            stream: true,
            prompt_tokens: self.usage_prompt,
            completion_tokens: tokens,
            energy_j,
            naive_energy_j: self.energy.joules_so_far(),
            joules_per_token: per_token(energy_j, tokens),
            ttft_ms: self.first_token_at.map(|t| (t - self.start).as_secs_f64() * 1000.0),
            total_ms: self.start.elapsed().as_secs_f64() * 1000.0,
            decode_tokens_per_s,
            completed: self.done,
        });
    }
}

fn per_token(energy_j: f64, tokens: u64) -> Option<f64> {
    (tokens > 0).then(|| energy_j / tokens as f64)
}
