//! Wattson gateway.
//!
//! Sits between clients and an inference server:
//!
//!     client  ->  wattson (:8080)  ->  vLLM / SGLang / mock (:8000)
//!
//! Step 2: forwards requests, streams responses back, measures tokens and
//! latency per request. Energy accounting comes in the next steps.
//!
//! Configuration (environment variables):
//!   WATTSON_PORT       port to listen on            (default 8080)
//!   WATTSON_UPSTREAM   inference server base URL    (default http://127.0.0.1:8000)

mod proxy;
mod stats;

use std::{
    env,
    net::SocketAddr,
    sync::{atomic::AtomicU64, Arc},
};

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};

use proxy::AppState;
use stats::{Stats, Summary};

async fn health() -> &'static str {
    "ok"
}

async fn stats_handler(State(state): State<AppState>) -> Json<Summary> {
    Json(state.stats.summary())
}

#[tokio::main]
async fn main() {
    let port: u16 = env::var("WATTSON_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let upstream = env::var("WATTSON_UPSTREAM")
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string())
        .trim_end_matches('/')
        .to_string();

    let state = AppState {
        client: reqwest::Client::new(),
        upstream: upstream.clone(),
        stats: Stats::default(),
        next_id: Arc::new(AtomicU64::new(1)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats_handler))
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("wattson listening on http://{addr}  ->  upstream {upstream}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
