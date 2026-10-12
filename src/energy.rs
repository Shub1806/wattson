//! Where GPU energy readings come from.
//!
//! Everything goes through the `EnergySource` trait, so the rest of the
//! gateway doesn't care whether the numbers are real or simulated:
//!
//!   - `SimulatedGpu` (this step): runs anywhere, including a Mac. It fakes
//!     a T4-like GPU whose power depends on what the gateway is doing.
//!   - `NvmlGpu` (later, on Kaggle): reads the real NVIDIA energy counter.
//!
//! Units follow NVML: energy in millijoules (mJ), power in milliwatts (mW).

use std::{
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

pub trait EnergySource: Send + Sync {
    /// Short name shown in /energy, e.g. "simulated" or "nvml".
    fn name(&self) -> &str;
    /// Total energy used since some fixed starting point. Only ever goes up,
    /// so the energy of any interval is `end - start`.
    fn total_energy_mj(&self) -> u64;
    /// Instantaneous power draw.
    fn power_mw(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Activity: how many requests are in each phase right now.
// ---------------------------------------------------------------------------

/// Live count of in-flight requests per phase. The proxy updates it; the
/// simulated GPU reads it to decide how much power to "draw".
#[derive(Default)]
pub struct Activity {
    prefilling: AtomicI64,
    decoding: AtomicI64,
}

impl Activity {
    pub fn prefilling(&self) -> i64 {
        self.prefilling.load(Ordering::Relaxed)
    }
    pub fn decoding(&self) -> i64 {
        self.decoding.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Phase {
    Prefill,
    Decode,
}

/// Marks one request as being in a phase. Moving between phases or dropping
/// the guard keeps the counters correct, even if the client disconnects.
pub struct PhaseGuard {
    activity: Arc<Activity>,
    phase: Phase,
}

impl PhaseGuard {
    pub fn new(activity: Arc<Activity>, phase: Phase) -> Self {
        counter(&activity, phase).fetch_add(1, Ordering::Relaxed);
        Self { activity, phase }
    }

    pub fn enter(&mut self, phase: Phase) {
        if phase != self.phase {
            counter(&self.activity, self.phase).fetch_sub(1, Ordering::Relaxed);
            counter(&self.activity, phase).fetch_add(1, Ordering::Relaxed);
            self.phase = phase;
        }
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        counter(&self.activity, self.phase).fetch_sub(1, Ordering::Relaxed);
    }
}

fn counter(activity: &Activity, phase: Phase) -> &AtomicI64 {
    match phase {
        Phase::Prefill => &activity.prefilling,
        Phase::Decode => &activity.decoding,
    }
}

// ---------------------------------------------------------------------------
// SimulatedGpu
// ---------------------------------------------------------------------------

/// A rough model of an NVIDIA T4 (70 W limit), good enough to develop with:
///
/// - **Idle** draws a baseline even with nothing running.
/// - **Prefill** processes the whole prompt in large matrix multiplies, which
///   is compute-bound, so it pushes power up a lot.
/// - **Decode** generates one token at a time and mostly waits on memory
///   (memory-bandwidth-bound), so it draws less. Extra concurrent decoding
///   requests are batched together and add less and less power each, which
///   is why batching lowers energy per token.
///
/// The real numbers come from NVML on Kaggle; this only has to behave sensibly.
pub struct SimulatedGpu {
    activity: Arc<Activity>,
    state: Mutex<SimState>,
}

struct SimState {
    last_update: Instant,
    energy_mj: f64,
}

const IDLE_W: f64 = 18.0;
const PREFILL_W: f64 = 40.0;
const DECODE_MAX_W: f64 = 30.0;
const POWER_LIMIT_W: f64 = 70.0;

impl SimulatedGpu {
    /// Creates the simulator and starts a background task that integrates
    /// power over time in small steps, like the real hardware counter does.
    pub fn start(activity: Arc<Activity>) -> Arc<Self> {
        let gpu = Arc::new(Self {
            activity,
            state: Mutex::new(SimState { last_update: Instant::now(), energy_mj: 0.0 }),
        });
        let ticker = gpu.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10));
            loop {
                interval.tick().await;
                ticker.total_energy_mj();
            }
        });
        gpu
    }

    fn power_w(&self) -> f64 {
        let prefilling = self.activity.prefilling().max(0);
        let decoding = self.activity.decoding().max(0);
        let prefill = if prefilling > 0 { PREFILL_W } else { 0.0 };
        // Saturating curve: 1 request ~9 W, 2 ~15 W, 4 ~23 W, many -> 30 W.
        let decode = DECODE_MAX_W * (1.0 - 0.7_f64.powi(decoding as i32));
        (IDLE_W + prefill + decode).min(POWER_LIMIT_W)
    }
}

impl EnergySource for SimulatedGpu {
    fn name(&self) -> &str {
        "simulated"
    }

    fn total_energy_mj(&self) -> u64 {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        let dt = (now - state.last_update).as_secs_f64();
        state.energy_mj += self.power_w() * dt * 1000.0; // W * s = J; *1000 = mJ
        state.last_update = now;
        state.energy_mj as u64
    }

    fn power_mw(&self) -> u64 {
        (self.power_w() * 1000.0) as u64
    }
}
