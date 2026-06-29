//! Model load + single-request generation, lifted from candle's
//! `quantized-qwen3` example and trimmed to a reusable, metrics-returning core.
//!
//! The model holds mutable KV-cache state, so [`Engine::generate`] resets it at
//! the start of every request — this is what makes a warm container safe to
//! reuse across invocations instead of reloading the GGUF each time.

use anyhow::{anyhow, Result};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use lambda_core::{GenMetrics, GenRequest, TextModel};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

pub struct Engine {
    model: Qwen3,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: u32,
}

impl TextModel for Engine {
    // Inherent `generate` (method resolution prefers it, so no recursion).
    fn generate(&mut self, req: &GenRequest) -> Result<GenMetrics> {
        self.generate(req)
    }
}

impl Engine {
    /// Load the GGUF weights and tokenizer onto the CPU. On CPU the Qwen3 model
    /// selects the interleaved raw-KV / flash path the perf branch tuned.
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| anyhow!("open {}: {e}", model_path.display()))?;
        let t_read = Instant::now();
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let gguf_read_ms = t_read.elapsed().as_millis();
        let t_build = Instant::now();
        let model = Qwen3::from_gguf(content, &mut file, &device)?;
        let model_build_ms = t_build.elapsed().as_millis();
        // Cold-start load breakdown (greppable in CloudWatch): read = GGUF bytes
        // into RAM, build = QTensor construction. Tells us which half to attack.
        eprintln!("COLDSTART gguf_read_ms={gguf_read_ms} model_build_ms={model_build_ms}");
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let eos_token = *tokenizer
            .get_vocab(true)
            .get("<|im_end|>")
            .ok_or_else(|| anyhow!("tokenizer is missing the <|im_end|> token"))?;
        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token,
        })
    }

    pub fn generate(&mut self, req: &GenRequest) -> Result<GenMetrics> {
        // Fresh conversation each invocation: drop any KV state left by the
        // previous warm request.
        self.model.clear_kv_cache();

        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            req.prompt
        );
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;
        let tokens = encoding.get_ids();
        if tokens.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }

        let mut logits_processor = {
            let temperature = req.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (req.top_k, req.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(req.seed, sampling)
        };

        // Prefill: one forward over the whole prompt.
        let t_prefill = Instant::now();
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        let prefill_dt = t_prefill.elapsed();

        let mut all_tokens = vec![next_token];
        let to_sample = req.max_tokens.saturating_sub(1);

        // Decode: one token per forward, append to KV cache via the position.
        let t_decode = Instant::now();
        let mut sampled = 0usize;
        if next_token != self.eos_token {
            for index in 0..to_sample {
                let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                let logits = self
                    .model
                    .forward(&input, tokens.len() + index)?
                    .squeeze(0)?;
                let logits = if req.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = all_tokens.len().saturating_sub(req.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        req.repeat_penalty,
                        &all_tokens[start_at..],
                    )?
                };
                next_token = logits_processor.sample(&logits)?;
                sampled += 1;
                if next_token == self.eos_token {
                    break;
                }
                all_tokens.push(next_token);
            }
        }
        let decode_dt = t_decode.elapsed();

        let text = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(anyhow::Error::msg)?;

        let prefill_s = prefill_dt.as_secs_f64();
        let decode_s = decode_dt.as_secs_f64();
        Ok(GenMetrics {
            text,
            prompt_tokens: tokens.len(),
            generated_tokens: sampled,
            prefill_ms: prefill_s * 1e3,
            decode_ms: decode_s * 1e3,
            prefill_tok_s: if prefill_s > 0. {
                tokens.len() as f64 / prefill_s
            } else {
                0.
            },
            decode_tok_s: if decode_s > 0. {
                sampled as f64 / decode_s
            } else {
                0.
            },
        })
    }
}
