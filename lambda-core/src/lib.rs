//! Model-agnostic serving harness for the model lambdas.
//!
//! A model crate implements [`TextModel`] and calls [`serve`] with a loader; the
//! harness owns the rest: warm-once lifecycle, a request mutex (Lambda runs one
//! request per container at a time), CPU work on `spawn_blocking`, JSON I/O, and
//! per-phase timing for the cost/token benchmark.
//!
//! Deliberately free of any ML dependency so the public/private split is clean —
//! a private model crate (here or in a separate private repo) depends on this
//! crate and nothing else changes.

use lambda_runtime::{service_fn, LambdaEvent};
pub use lambda_runtime::Error;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// A single generation request, already parsed and defaulted.
pub struct GenRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

/// Result of one generation, with per-phase timing for cost/token analysis.
pub struct GenMetrics {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
}

/// Implemented by each model crate. Must reset any per-conversation state (e.g.
/// the KV cache) at the start of `generate`, since the instance is reused across
/// warm invocations.
pub trait TextModel: Send {
    fn generate(&mut self, req: &GenRequest) -> anyhow::Result<GenMetrics>;
}

/// Builds the model. A plain fn pointer so it can initialize a `'static` slot.
pub type Loader = fn() -> anyhow::Result<Box<dyn TextModel>>;

static ENGINE: OnceLock<Mutex<Box<dyn TextModel>>> = OnceLock::new();
static FIRST_INVOCATION: AtomicBool = AtomicBool::new(true);
static LOAD_MS: AtomicU64 = AtomicU64::new(0);

/// Read a filesystem path from `var`, falling back to `default`. Lets each model
/// keep a sensible local-dev default while the container overrides via env.
pub fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var(var).unwrap_or_else(|_| default.to_string()).into()
}

#[derive(Deserialize)]
struct Request {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f64,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: u64,
    #[serde(default = "default_repeat_penalty")]
    repeat_penalty: f32,
    #[serde(default = "default_repeat_last_n")]
    repeat_last_n: usize,
}

fn default_max_tokens() -> usize {
    256
}
fn default_repeat_penalty() -> f32 {
    1.1
}
fn default_repeat_last_n() -> usize {
    64
}

#[derive(Serialize)]
struct Response {
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    /// True only for the invocation that paid the model-load cost.
    cold_start: bool,
    /// Model load time in ms (0 on warm invocations).
    load_ms: u64,
}

fn warm(loader: Loader) {
    ENGINE.get_or_init(|| {
        let t = Instant::now();
        let model = loader().expect("failed to load model");
        LOAD_MS.store(t.elapsed().as_millis() as u64, Ordering::Relaxed);
        Mutex::new(model)
    });
}

async fn handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let req = event.payload;
    let gen = GenRequest {
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repeat_penalty: req.repeat_penalty,
        repeat_last_n: req.repeat_last_n,
    };

    // Heavy CPU work off the async reactor.
    let (result, cold_start) = tokio::task::spawn_blocking(move || {
        let engine = ENGINE.get().expect("engine not initialized");
        let mut guard = engine.lock().expect("engine mutex poisoned");
        let cold_start = FIRST_INVOCATION.swap(false, Ordering::Relaxed);
        (guard.generate(&gen), cold_start)
    })
    .await?;

    let m = result?;
    Ok(Response {
        text: m.text,
        prompt_tokens: m.prompt_tokens,
        generated_tokens: m.generated_tokens,
        prefill_ms: m.prefill_ms,
        decode_ms: m.decode_ms,
        prefill_tok_s: m.prefill_tok_s,
        decode_tok_s: m.decode_tok_s,
        cold_start,
        load_ms: if cold_start {
            LOAD_MS.load(Ordering::Relaxed)
        } else {
            0
        },
    })
}

/// Entry point for a model lambda: initialize logging, warm the model during
/// init, then serve. Call from the model crate's `main`.
pub async fn serve(loader: Loader) -> Result<(), Error> {
    tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    // Warm during init so the first request isn't charged for the load.
    warm(loader);

    lambda_runtime::run(service_fn(handler)).await
}
