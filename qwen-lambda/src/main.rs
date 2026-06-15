//! Qwen3-0.6B Q4_K_M lambda: wires the model into the shared serving harness.
//! All runtime/lifecycle/IO lives in `lambda-core`; this crate only builds the
//! model and hands `serve` a loader.

mod engine;

use engine::Engine;
use lambda_core::{env_path, serve, Error, TextModel};

fn load_model() -> anyhow::Result<Box<dyn TextModel>> {
    let model = env_path("MODEL_PATH", "/opt/model/Qwen3-0.6B-Q4_K_M.gguf");
    let tokenizer = env_path("TOKENIZER_PATH", "/opt/model/tokenizer.json");
    Ok(Box::new(Engine::load(&model, &tokenizer)?))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    serve(load_model).await
}
