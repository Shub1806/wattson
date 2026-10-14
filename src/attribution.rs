//! Fair energy attribution: who used which joules?
//!
//! The GPU only gives us one number: total energy so far. When several
//! requests run at once, we have to decide how to split it. The `Ledger`
//! does this in small time slices:
//!
//!   every ~20 ms (and whenever a request starts, changes phase or ends):
//!     delta = energy counter now - energy counter at the last slice
//!     if no request is active  -> delta goes to the "idle" bucket
//!     otherwise                -> delta is split between active requests,
//!                                 in proportion to their phase weight
//!
//! So every joule is counted exactly once: the per-request totals plus idle
//! always add up to what the GPU really used.
//!
//! Phase weights: a request in prefill (compute-bound, processing the whole
//! prompt) is charged more per slice than one in decode (memory-bound, one
//! token at a time). The weights below are a starting assumption; measuring
//! prefill vs decode power on a real GPU lets us calibrate them.
//!
//! Queued requests: before its first token, a request is either really
//! prefilling or just waiting in the server's queue, and the gateway can't
//! tell which. If the server publishes its queue length (vLLM's
//! `vllm:num_requests_waiting`), we use it: if N of the M not-yet-started
//! requests are waiting, only (M - N) of them are really prefilling. We
//! can't know *which* ones (requests arriving together may reach the server
//! in any order), so each of the M gets (M - N) / M of the prefill weight.
//! That is right on average and never charges a waiting request in full.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use serde::Serialize;

use crate::energy::{Activity, EnergySource, Phase, PhaseGuard};

const PREFILL_WEIGHT: f64 = 4.0;
const DECODE_WEIGHT: f64 = 1.0;
const SLICE: Duration = Duration::from_millis(20);

pub struct Ledger {
    source: Arc<dyn EnergySource>,
    inner: Mutex<LedgerInner>,
    /// Requests waiting in the server's queue, from its /metrics endpoint.
    queued: AtomicUsize,
}

struct LedgerInner {
    last_mj: u64,
    active: HashMap<u64, Entry>,
    idle_j: f64,
    attributed_j: f64,
}

struct Entry {
    phase: Phase,
    joules: f64,
}

#[derive(Serialize)]
pub struct LedgerTotals {
    /// Energy charged to requests.
    pub attributed_j: f64,
    /// Energy used while no request was running.
    pub idle_j: f64,
    pub active_requests: usize,
    /// Server-reported queue length (0 if the server doesn't publish it).
    pub server_queue: usize,
}

impl Ledger {
    /// Creates the ledger and starts the background task that closes a
    /// time slice every 20 ms.
    pub fn start(source: Arc<dyn EnergySource>) -> Arc<Self> {
        let last_mj = source.total_energy_mj();
        let ledger = Arc::new(Self {
            source,
            inner: Mutex::new(LedgerInner {
                last_mj,
                active: HashMap::new(),
                idle_j: 0.0,
                attributed_j: 0.0,
            }),
            queued: AtomicUsize::new(0),
        });
        let ticker = ledger.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(SLICE);
            loop {
                interval.tick().await;
                let mut inner = ticker.inner.lock().unwrap();
                ticker.close_slice(&mut inner);
            }
        });
        ledger
    }

    /// Splits the energy used since the last slice among active requests.
    /// Called with the lock held, so slices never overlap.
    fn close_slice(&self, inner: &mut LedgerInner) {
        let now_mj = self.source.total_energy_mj();
        let delta_j = now_mj.saturating_sub(inner.last_mj) as f64 / 1000.0;
        inner.last_mj = now_mj;

        // Share of not-yet-started requests that are really prefilling.
        let not_started = inner.active.values().filter(|e| e.phase == Phase::Prefill).count();
        let queued = self.queued.load(Ordering::Relaxed).min(not_started);
        let running_share = if not_started == 0 {
            1.0
        } else {
            (not_started - queued) as f64 / not_started as f64
        };
        let slice_weight = |e: &Entry| match e.phase {
            Phase::Prefill => PREFILL_WEIGHT * running_share,
            Phase::Decode => DECODE_WEIGHT,
        };

        let total_weight: f64 = inner.active.values().map(slice_weight).sum();
        if total_weight == 0.0 {
            inner.idle_j += delta_j;
            return;
        }
        for entry in inner.active.values_mut() {
            entry.joules += delta_j * slice_weight(entry) / total_weight;
        }
        inner.attributed_j += delta_j;
    }

    // Every change closes the current slice first, so energy used *before*
    // the change is split using the situation *before* the change.

    fn register(&self, id: u64, phase: Phase) {
        let mut inner = self.inner.lock().unwrap();
        self.close_slice(&mut inner);
        inner.active.insert(id, Entry { phase, joules: 0.0 });
    }

    fn set_phase(&self, id: u64, phase: Phase) {
        let mut inner = self.inner.lock().unwrap();
        self.close_slice(&mut inner);
        if let Some(entry) = inner.active.get_mut(&id) {
            entry.phase = phase;
        }
    }

    fn finish(&self, id: u64) -> f64 {
        let mut inner = self.inner.lock().unwrap();
        self.close_slice(&mut inner);
        inner.active.remove(&id).map(|e| e.joules).unwrap_or(0.0)
    }

    pub fn totals(&self) -> LedgerTotals {
        let inner = self.inner.lock().unwrap();
        LedgerTotals {
            attributed_j: inner.attributed_j,
            idle_j: inner.idle_j,
            active_requests: inner.active.len(),
            server_queue: self.queued.load(Ordering::Relaxed),
        }
    }

    /// Polls `{upstream}/metrics` every 50 ms for the server's queue length.
    /// Servers without that metric (like the mock server) are simply ignored.
    pub fn watch_server_queue(self: &Arc<Self>, client: reqwest::Client, upstream: String) {
        let ledger = self.clone();
        tokio::spawn(async move {
            let url = format!("{upstream}/metrics");
            let announced = AtomicBool::new(false);
            let mut interval = tokio::time::interval(Duration::from_millis(50));
            loop {
                interval.tick().await;
                let Ok(resp) = client.get(&url).send().await else { continue };
                let Ok(text) = resp.text().await else { continue };
                if let Some(waiting) = parse_waiting(&text) {
                    if !announced.swap(true, Ordering::Relaxed) {
                        println!("queue metrics: using vllm:num_requests_waiting from {url}");
                    }
                    ledger.queued.store(waiting, Ordering::Relaxed);
                }
            }
        });
    }
}

/// Sums `vllm:num_requests_waiting` across all label sets in Prometheus text.
fn parse_waiting(metrics: &str) -> Option<usize> {
    let values: Vec<f64> = metrics
        .lines()
        .filter(|line| line.starts_with("vllm:num_requests_waiting"))
        .filter_map(|line| line.split_whitespace().last()?.parse().ok())
        .collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>().round() as usize)
}

/// One request's meter. Keeps the ledger (who gets charged) and the GPU
/// activity counters (what the simulated GPU sees) in sync.
pub struct Meter {
    id: u64,
    ledger: Arc<Ledger>,
    phase: PhaseGuard,
    finished: bool,
}

impl Meter {
    pub fn start(ledger: Arc<Ledger>, activity: Arc<Activity>, id: u64, phase: Phase) -> Self {
        ledger.register(id, phase);
        Self { id, ledger, phase: PhaseGuard::new(activity, phase), finished: false }
    }

    pub fn enter(&mut self, phase: Phase) {
        self.ledger.set_phase(self.id, phase);
        self.phase.enter(phase);
    }

    /// Stops metering and returns this request's fair share of energy (J).
    pub fn finish(&mut self) -> f64 {
        self.finished = true;
        self.ledger.finish(self.id)
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        if !self.finished {
            self.ledger.finish(self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_waiting;

    #[test]
    fn parses_vllm_queue_metric() {
        let text = "# HELP vllm:num_requests_waiting Number of requests waiting.\n\
                    # TYPE vllm:num_requests_waiting gauge\n\
                    vllm:num_requests_waiting{engine=\"0\",model_name=\"qwen\"} 3.0\n\
                    vllm:num_requests_running{engine=\"0\",model_name=\"qwen\"} 2.0\n";
        assert_eq!(parse_waiting(text), Some(3));
    }

    #[test]
    fn sums_across_label_sets_and_ignores_other_servers() {
        let text = "vllm:num_requests_waiting{engine=\"0\"} 1\nvllm:num_requests_waiting{engine=\"1\"} 2\n";
        assert_eq!(parse_waiting(text), Some(3));
        assert_eq!(parse_waiting("not found"), None);
    }
}
