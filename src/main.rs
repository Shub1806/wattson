//! Wattson gateway.
//!
//! Sits between clients and an inference server:
//!
//!     client  ->  wattson (:8080)  ->  vLLM / SGLang / mock (:8000)
//!
//! Forwards requests, streams responses back, and measures tokens, latency
//! and GPU energy per request.
//!
//! Configuration (environment variables):
//!   WATTSON_PORT       port to listen on            (default 8080)
//!   WATTSON_UPSTREAM   inference server base URL    (default http://127.0.0.1:8000)
//!   WATTSON_ENERGY     energy source: "sim" or "nvml"  (default sim)
//!   WATTSON_GPU_INDEX  which GPU to read with nvml     (default 0)

mod attribution;
mod energy;
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

use attribution::{Ledger, LedgerTotals};
use energy::{Activity, EnergySource, SimulatedGpu};
use proxy::AppState;
use serde::Serialize;
use stats::{Stats, Summary};

async fn health() -> &'static str {
    "ok"
}

async fn requests_handler(State(state): State<AppState>) -> Json<Vec<stats::RequestRecord>> {
    Json(state.stats.all())
}

async fn stats_handler(State(state): State<AppState>) -> Json<Summary> {
    Json(state.stats.summary())
}

#[derive(Serialize)]
struct EnergyStatus {
    source: String,
    power_w: f64,
    total_energy_j: f64,
    requests_prefilling: i64,
    requests_decoding: i64,
    /// How the GPU's energy has been split since the gateway started.
    ledger: LedgerTotals,
}

async fn energy_handler(State(state): State<AppState>) -> Json<EnergyStatus> {
    Json(EnergyStatus {
        source: state.energy.name().to_string(),
        power_w: state.energy.power_mw() as f64 / 1000.0,
        total_energy_j: state.energy.total_energy_mj() as f64 / 1000.0,
        requests_prefilling: state.activity.prefilling(),
        requests_decoding: state.activity.decoding(),
        ledger: state.ledger.totals(),
    })
}

#[tokio::main]
async fn main() {
    let port: u16 = env::var("WATTSON_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let upstream = env::var("WATTSON_UPSTREAM")
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string())
        .trim_end_matches('/')
        .to_string();

    let activity = Arc::new(Activity::default());
    let energy: Arc<dyn EnergySource> = match env::var("WATTSON_ENERGY").as_deref() {
        Ok("sim") | Err(_) => SimulatedGpu::start(activity.clone()),
        #[cfg(target_os = "linux")]
        Ok("nvml") => {
            let index = env::var("WATTSON_GPU_INDEX").ok().and_then(|i| i.parse().ok()).unwrap_or(0);
            match energy::NvmlGpu::start(index) {
                Ok(gpu) => gpu,
                Err(err) => {
                    eprintln!("error: {err}\nhint: use WATTSON_ENERGY=sim on machines without an NVIDIA GPU");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        Ok("nvml") => {
            eprintln!("error: WATTSON_ENERGY=nvml needs Linux with an NVIDIA GPU (e.g. Kaggle); use sim here");
            std::process::exit(1);
        }
        Ok(other) => {
            eprintln!("error: unknown WATTSON_ENERGY={other:?} (use \"sim\" or \"nvml\")");
            std::process::exit(1);
        }
    };

    let client = reqwest::Client::new();
    let ledger = Ledger::start(energy.clone());
    ledger.watch_server_queue(client.clone(), upstream.clone());

    let state = AppState {
        client,
        upstream: upstream.clone(),
        stats: Stats::default(),
        next_id: Arc::new(AtomicU64::new(1)),
        energy: energy.clone(),
        activity,
        ledger,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats_handler))
        .route("/energy", get(energy_handler))
        .route("/requests", get(requests_handler))
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!(
        "wattson listening on http://{addr}  ->  upstream {upstream}  (energy: {})",
        energy.name()
    );
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
